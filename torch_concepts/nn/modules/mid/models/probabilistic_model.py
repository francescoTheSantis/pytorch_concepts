"""
Probabilistic graphical model abstractions for concept-based networks.

This module defines:

* :class:`_ProbabilisticModelBase` — abstract base class for all graphical
  model types.  Provides shared infrastructure (variable lookup, CPD
  registration, parent-input building) and defines the extension points that
  each concrete model type must implement: :meth:`_initialize`,
  :meth:`forward`, and :meth:`query`.

* :class:`BayesianNetwork` — a Pyro-backed directed graphical model (DAG).
  This is the primary concrete implementation.  All existing inference engines
  (:class:`DeterministicInference`, :class:`AncestralSamplingInference`,
  :class:`ELBOInference`, …) are compatible with it since it exposes the
  same interface (``variables``, ``factors``, ``get_module_of_concept``, etc.).

* :data:`ProbabilisticModel` — backward-compatible alias for
  :class:`BayesianNetwork`.  Existing code that instantiates
  ``ProbabilisticModel(variables, factors)`` continues to work unchanged.

Planned extensions (not yet implemented)
-----------------------------------------
The following concrete subclasses are on the roadmap:

* ``MarkovRandomField`` — undirected model backed by unnormalised potential
  functions (``pyro.factor`` sites).  Supports MCMC and belief propagation;
  ancestral sampling does not apply.

* ``ChainGraph`` — partially directed model mixing directed CPD edges
  (``pyro.sample``) with undirected cliques (potential functions).  Useful
  for hybrid causal-Markov models.

Both will follow the same three-method extension pattern as
:class:`BayesianNetwork`, subclassing :class:`_ProbabilisticModelBase`.
"""
from __future__ import annotations

import copy
import inspect
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch.distributions import Distribution

import pyro
import pyro.distributions as pydist
import pyro.poutine as poutine
from pyro.nn import PyroModule

from .variable import Variable, ConceptVariable, ExogenousVariable, LatentVariable
from .cpd import ParametricCPD


# ---------------------------------------------------------------------------
# Helper: topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: concat empirical means from posterior samples
# ---------------------------------------------------------------------------

def _concat_sample_means(
    variables: List[str],
    samples: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Concatenate the per-variable mean over the leading sample dim.

    Used by sample-based engines (importance / exact-discrete / mcmc /
    ancestral) to populate ``InferenceOutput.probs`` in the same
    ``(B, sum_sizes)`` layout produced by :class:`ForwardInference`.

    Returns ``None`` when no requested variable has stackable samples.
    """
    parts: List[torch.Tensor] = []
    for var in variables:
        t = samples.get(var)
        if t is None or t.dim() < 1:
            continue
        parts.append(t.float().mean(0))
    if not parts:
        return None
    try:
        return torch.cat(parts, dim=-1)
    except RuntimeError:
        # Inconsistent batch shapes across sites — leave probs unset.
        return None


# ---------------------------------------------------------------------------
# Helper: topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def _topological_sort(variables: List[Variable], get_parents) -> List[Variable]:
    """Return variables in topological order."""
    in_degree = {v.concept: 0 for v in variables}
    adj: Dict[str, List[str]] = {v.concept: [] for v in variables}
    var_map = {v.concept: v for v in variables}

    for var in variables:
        for pv in get_parents(var.concept):
            adj[pv.concept].append(var.concept)
            in_degree[var.concept] += 1

    queue = [var_map[n] for n, d in in_degree.items() if d == 0]
    ordered: List[Variable] = []

    while queue:
        var = queue.pop(0)
        ordered.append(var)
        for child_name in adj[var.concept]:
            in_degree[child_name] -= 1
            if in_degree[child_name] == 0:
                queue.append(var_map[child_name])

    return ordered


# ---------------------------------------------------------------------------
# _ProbabilisticModelBase — abstract base
# ---------------------------------------------------------------------------

class _ProbabilisticModelBase(PyroModule):
    """
    Abstract base for concept-based probabilistic graphical models.

    Subclasses must implement :meth:`_initialize`, :meth:`forward`, and
    :meth:`query`.  All shared infrastructure (variable lookup, CPD
    registration, parent-input building) is provided here.

    To add a new model type create a subclass and override the three
    extension points::

        class MyModel(_ProbabilisticModelBase):
            def _initialize(self, factors): ...
            def forward(self, evidence, targets=None): ...
            def query(self, variables, evidence, **kw): ...

    Parameters
    ----------
    variables : List[Variable]
        All concept/latent/exogenous variables in the model.
    factors : list
        One factor per variable.

    Attributes
    ----------
    variables : List[Variable]
        All variables.
    factors : nn.ModuleDict
        Concept-name → factor mapping.
    concept_to_variable : Dict[str, Variable]
        Name → variable lookup.
    """

    def __init__(
        self,
        variables: List[Variable],
        factors: list,
    ) -> None:
        super().__init__()
        self.variables: List[Variable] = variables
        self.factors: nn.ModuleDict = nn.ModuleDict()
        self.concept_to_variable: Dict[str, Variable] = {
            var.concept: var for var in variables
        }
        self._shared_cpd_map: Dict[str, str] = {}
        self._initialize(factors)

    # ------------------------------------------------------------------
    # Extension points (subclasses must override)
    # ------------------------------------------------------------------

    def _initialize(self, factors: list) -> None:
        """Subclass-specific factor registration and graph setup."""
        raise NotImplementedError

    def forward(
        self,
        evidence: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generative forward pass (model-specific)."""
        raise NotImplementedError

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Posterior-predictive query (model-specific)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared infrastructure
    # ------------------------------------------------------------------

    def _register_shared_cpd(self, factor: ParametricCPD) -> None:
        """Register a shared CPD and map secondary concept names to it."""
        shared_name = getattr(factor, 'shared_name', None)
        key = shared_name if shared_name else factor.concept
        factor.variable = self.concept_to_variable.get(factor.concept)
        self.factors[str(key)] = factor
        for name in factor.concepts:
            if name != key:
                self._shared_cpd_map[name] = key

    def _resolve_parent_refs(self, parents: list) -> List[Variable]:
        """Resolve a mixed list of Variable / str references to Variable objects."""
        resolved = []
        for ref in parents:
            if isinstance(ref, str):
                if ref not in self.concept_to_variable:
                    raise ValueError(f"Parent concept '{ref}' not found in any variable.")
                resolved.append(self.concept_to_variable[ref])
            elif isinstance(ref, Variable):
                resolved.append(ref)
            elif hasattr(ref, 'concept'):
                resolved.append(self.concept_to_variable[ref.concept])
            else:
                raise TypeError(f"Invalid parent reference type: {type(ref)}")
        # Deduplicate while preserving order
        return list({id(p): p for p in resolved}.values())

    # ------------------------------------------------------------------
    # Public API helpers
    # ------------------------------------------------------------------

    @property
    def parametric_cpds(self) -> nn.ModuleDict:
        """Alias for ``self.factors``."""
        return self.factors

    def get_module_of_concept(self, concept_name: str) -> Optional[ParametricCPD]:
        """Return the CPD for *concept_name*, or ``None``."""
        if str(concept_name) in self.factors:
            return self.factors[str(concept_name)]
        if concept_name in self._shared_cpd_map:
            return self.factors[str(self._shared_cpd_map[concept_name])]
        return None

    def get_variable_parents(self, concept_name: str) -> List[Variable]:
        """Return the parent variables of a concept."""
        cpd = self.get_module_of_concept(concept_name)
        return cpd.parents if cpd is not None else []

    def get_by_distribution(self, distribution_class: Type[Distribution]) -> List[Variable]:
        """Return all variables with a given distribution type."""
        return [v for v in self.variables if v.distribution is distribution_class]

    def _make_temp_parametric_cpd(self, concept: str, module: nn.Module) -> ParametricCPD:
        """Create a temporary ParametricCPD for table-building helpers."""
        if isinstance(module, ParametricCPD):
            parametrization = module.parametrization
        else:
            parametrization = module
        f = ParametricCPD(concept=concept, parametrization=parametrization)
        f.variable = self.concept_to_variable[concept]
        stored = self.factors[str(concept)] if str(concept) in self.factors else None
        f.parents = stored.parents if stored is not None else []
        return f

    # ------------------------------------------------------------------
    # Parent-input builder (shared by forward, guide, and inference engines)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_parametrization_sig(cpd: ParametricCPD) -> inspect.Signature:
        """Return the forward signature of the underlying parametrization module."""
        if isinstance(cpd.parametrization, nn.ModuleDict):
            first_mod = next(iter(cpd.parametrization.values()))
            return inspect.signature(first_mod.forward)
        return inspect.signature(cpd.parametrization.forward)

    @staticmethod
    def _build_parent_kwargs(
        cpd: ParametricCPD,
        context: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Build the ``**kwargs`` dict for ``cpd.forward()``."""
        parent_input: List[torch.Tensor] = []
        parent_concepts: List[torch.Tensor] = []

        for pv in cpd.parents:
            val = context[pv.concept]
            if isinstance(pv, ConceptVariable):
                parent_concepts.append(val)
            else:
                parent_input.append(val)

        try:
            sig = _ProbabilisticModelBase._get_parametrization_sig(cpd)
            pnames = set(sig.parameters.keys())
        except (ValueError, TypeError):
            pnames = set()

        # PyC-style layers (concepts / latent / exogenous keyword args)
        if 'concepts' in pnames or 'latent' in pnames or 'exogenous' in pnames:
            kwargs: Dict[str, torch.Tensor] = {}
            if 'concepts' in pnames and parent_concepts:
                kwargs['concepts'] = torch.cat(parent_concepts, dim=-1)
            if 'latent' in pnames and parent_input:
                kwargs['latent'] = torch.cat(parent_input, dim=-1)
            elif 'exogenous' in pnames and parent_input:
                kwargs['exogenous'] = torch.cat(parent_input, dim=-1)
            return kwargs

        # Standard module: concatenate everything → first positional param
        all_vals = parent_concepts + parent_input
        # Align ndim before cat (handles Predictive parallel=True shape mismatch)
        if all_vals:
            max_ndim = max(v.dim() for v in all_vals)
            padded = [
                v.reshape(*([1] * (max_ndim - v.dim())), *v.shape)
                for v in all_vals
            ]
            leading = torch.broadcast_shapes(*[v.shape[:-1] for v in padded])
            all_vals = [v.expand(*leading, v.shape[-1]) for v in padded]
        combined = torch.cat(all_vals, dim=-1)
        try:
            sig = _ProbabilisticModelBase._get_parametrization_sig(cpd)
            first = next(iter(sig.parameters))
            return {first: combined}
        except StopIteration:
            return {'input': combined}

    @staticmethod
    def _run_cpd(
        cpd: ParametricCPD,
        context: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run ``cpd.forward()`` and return the raw params tensor."""
        if not cpd.parents:
            # Root node: evidence[concept] → transform (usually Identity)
            key = getattr(cpd, 'shared_name', None) or cpd.concept
            raw = evidence.get(key)
            if raw is None:
                raw = evidence.get(cpd.concept)
            if raw is None:
                raise ValueError(
                    f"Root variable '{cpd.concept}' not found in evidence dict."
                )
            return cpd(raw)          # cpd.forward(raw) handles dict/single form

        kwargs = _ProbabilisticModelBase._build_parent_kwargs(cpd, context)
        return cpd(**kwargs)         # cpd.forward(**kwargs) handles dict/single form

    # ------------------------------------------------------------------
    # Shared per-variable propagation step (used by both
    # BayesianNetwork.forward and ForwardInference activations — see
    # design note 13.1 in the mid-level API deep-dive notebook)
    # ------------------------------------------------------------------

    @staticmethod
    def _propagate_raw(
        var: Variable,
        raw_params,
        mode: str = 'pyro',
        name: Optional[str] = None,
        obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert raw CPD parameters into a propagated value.

        Single source of truth for the per-variable propagation step shared
        by :meth:`BayesianNetwork.forward` (Pyro generative model) and
        ``ForwardInference.activate`` (deterministic / ancestral inference).

        Parameters
        ----------
        var : Variable
            The node being computed.
        raw_params : Tensor or Dict[str, Tensor]
            Raw output of the CPD's ``forward()``.
        mode : {'pyro', 'deterministic', 'ancestral'}
            * ``'pyro'`` — register a ``pyro.sample`` site (or
              ``pyro.deterministic`` for Delta variables).  Requires *name*
              and may use *obs* to condition.
            * ``'deterministic'`` — return the distribution's analytical
              mean (with sigmoid/softmax fallback for relaxed Bernoulli /
              Categorical).
            * ``'ancestral'`` — draw a sample (``rsample`` if available).
        name : str, optional
            Pyro site name (required for ``mode='pyro'``).
        obs : Tensor, optional
            Observed value to condition on (only used for ``mode='pyro'``).

        Returns
        -------
        Tensor
            The propagated value for this variable.
        """
        if mode == 'pyro':
            if var.is_deterministic:
                value = raw_params['value'] if isinstance(raw_params, dict) else raw_params
                return pyro.deterministic(name, value)
            d = var.make_distribution(raw_params)
            return pyro.sample(name, d, obs=obs)

        if mode == 'deterministic':
            from torch.distributions import (
                Bernoulli, RelaxedBernoulli,
                OneHotCategorical, RelaxedOneHotCategorical,
            )
            dist_cls = var.distribution
            # Bernoulli-family: sigmoid (works for RelaxedBernoulli too,
            # which has no closed-form mean).
            if dist_cls in (Bernoulli, RelaxedBernoulli):
                logits = raw_params['logits'] if isinstance(raw_params, dict) else raw_params
                return torch.sigmoid(logits)
            # Categorical-family: softmax.
            if dist_cls in (OneHotCategorical, RelaxedOneHotCategorical):
                logits = raw_params['logits'] if isinstance(raw_params, dict) else raw_params
                return torch.softmax(logits, dim=-1)
            # Otherwise, analytical mean.
            return var.make_distribution(raw_params).mean

        if mode == 'ancestral':
            d = var.make_distribution(raw_params)
            sample = d.rsample() if d.has_rsample else d.sample()
            if sample.dim() == 1:
                sample = sample.unsqueeze(-1)
            return sample

        raise ValueError(
            f"Unknown propagation mode '{mode}'. "
            f"Expected one of: 'pyro', 'deterministic', 'ancestral'."
        )


# ---------------------------------------------------------------------------
# BayesianNetwork — Pyro-backed directed graphical model (DAG)
# ---------------------------------------------------------------------------

class BayesianNetwork(_ProbabilisticModelBase):
    """
    Pyro-backed concept-based Bayesian Network (directed acyclic graph).

    Stores a set of :class:`Variable` nodes and :class:`ParametricCPD`
    factors.  All CPD parameters are tracked as standard ``nn.Module``
    parameters (requires ``pyro.settings.set(module_local_params=True)``).

    Compatible with :class:`DeterministicInference`,
    :class:`AncestralSamplingInference`, :class:`ELBOInference`, and all
    other inference engines that expect the standard
    :class:`ProbabilisticModel` interface.

    Attributes
    ----------
    sorted_variables : List[Variable]
        Variables in topological order (used by forward and guides).
    """

    _is_directed: bool = True

    def _initialize(self, factors: List[ParametricCPD]) -> None:
        """Register directed factors and compute topological order."""
        self._initialize_directed(factors)
        # Resolve any string parent references to Variable objects (post-registration)
        for cpd in self.factors.values():
            cpd.parents = self._resolve_parent_refs(cpd.parents)
        self.sorted_variables: List[Variable] = _topological_sort(
            self.variables,
            lambda name: self.factors[name].parents if name in self.factors else [],
        )

    def _initialize_directed(self, input_factors: List[ParametricCPD]) -> None:
        """Directed-model initialisation: lazy constructors + parent resolution."""
        from ...low.lazy import LazyConstructor

        for cpd in input_factors:
            if getattr(cpd, 'shared', False):
                if isinstance(cpd.parametrization, LazyConstructor):
                    raise NotImplementedError(
                        "LazyConstructor is not supported with shared=True CPDs.")
                self._register_shared_cpd(cpd)
                continue

            concept = cpd.concept
            assert isinstance(concept, str)

            if concept in self.concept_to_variable:
                cpd.variable = self.concept_to_variable[concept]

            if isinstance(cpd.parametrization, LazyConstructor):
                parent_vars = self._resolve_parent_refs(cpd.parents)
                in_concepts = in_exogenous = in_latent = 0
                for pv in parent_vars:
                    if isinstance(pv, ExogenousVariable):
                        in_exogenous = pv.size
                    elif isinstance(pv, ConceptVariable):
                        in_concepts += pv.size
                    else:
                        in_latent += pv.size

                out_concepts = (1 if isinstance(cpd.variable, ExogenousVariable)
                                else self.concept_to_variable[concept].size)

                initialized_layer = cpd.parametrization.build(
                    in_latent=in_latent,
                    in_concepts=in_concepts,
                    in_exogenous=in_exogenous,
                    out_concepts=out_concepts,
                )
                new_cpd = ParametricCPD(
                    concept=concept,
                    parametrization=initialized_layer,
                    parents=cpd.parents,
                )
                new_cpd.variable = cpd.variable
                cpd = new_cpd

            self.factors[str(concept)] = cpd

    # ------------------------------------------------------------------
    # Pyro generative model (forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        evidence: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Pyro generative model ``p(variables | evidence)``.

        Iterates over variables in topological order, runs each CPD, and
        records ``pyro.sample`` (or ``pyro.deterministic``) sites.

        Parameters
        ----------
        evidence : dict
            Maps variable names to observed tensors.  Root variables (e.g.
            ``'input'``) must always be present.
        targets : dict, optional
            Additional observed tensors (concept labels, task labels).
            Variables present here are conditioned on; absent variables are
            marginalised.

        Returns
        -------
        dict
            Maps variable names to sampled / observed tensors.
        """
        obs_dict: Dict[str, torch.Tensor] = (
            {**evidence, **targets} if targets else evidence
        )

        batch_size = next(iter(obs_dict.values())).shape[0]
        context: Dict[str, torch.Tensor] = {}

        with pyro.plate('data', batch_size):
            for var in self.sorted_variables:
                name = var.concept
                cpd = self.get_module_of_concept(name)
                if cpd is None:
                    continue

                params = self._run_cpd(cpd, context, obs_dict)
                context[name] = self._propagate_raw(
                    var, params,
                    mode='pyro',
                    name=name,
                    obs=obs_dict.get(name),
                )

        return context

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, torch.Tensor],
        num_samples: int = 100,
        guide=None,
        method: Optional[str] = None,
        **query_kwargs,
    ) -> 'InferenceOutput':
        """
        Posterior-predictive query.

        Parameters
        ----------
        variables : list of str
            Variable names to include in the returned samples.
        evidence : dict
            Observed values.  Root variables must always be present.
        num_samples : int
            Number of posterior samples to draw.
        guide : PyroModule, optional
            Variational guide for sampling.  If ``None``, ancestral (prior)
            sampling is used.
        method : str, optional
            Override the sampling algorithm.  Choices:
            ``"ancestral"``, ``"importance"``, ``"nuts"``, ``"hmc"``,
            ``"exact_discrete"``.
        **query_kwargs
            Extra kwargs forwarded to specialised algorithms (e.g.
            ``warmup_steps``, ``num_chains``, ``temperature``).

        Returns
        -------
        InferenceOutput
            ``result.samples`` maps variable name → Tensor of shape
            ``(num_samples, *batch_dims, variable_size)``.  ``result.probs``
            is the empirical mean over the leading sample dimension,
            concatenated in the requested-variable order (when shapes are
            consistent).
        """
        from ..inference.importance import ImportanceQuery
        from ..inference.mcmc import MCMCQuery
        from ..inference.exact_discrete import ExactDiscreteQuery
        from ...outputs import InferenceOutput

        _valid_methods = {
            "ancestral", "importance", "nuts", "hmc", "exact_discrete",
        }

        if method is not None and method not in _valid_methods:
            raise ValueError(
                f"Unknown method '{method}'. Valid: {sorted(_valid_methods)}"
            )

        if method == "importance":
            return ImportanceQuery(self).query(variables, evidence, num_samples)

        if method in ("nuts", "hmc"):
            return MCMCQuery(
                self,
                kernel=method,
                warmup_steps=query_kwargs.get("warmup_steps", 200),
                num_chains=query_kwargs.get("num_chains", 1),
                step_size=query_kwargs.get("step_size", 0.1),
                num_steps=query_kwargs.get("num_steps", 10),
            ).query(variables, evidence, num_samples)

        if method == "exact_discrete":
            return ExactDiscreteQuery(
                self,
                temperature=query_kwargs.get("temperature", 1),
            ).query(variables, evidence, num_samples)

        # Default: ancestral (no guide) or guide-based Predictive
        if guide is None or method == "ancestral":
            conditioned = poutine.condition(self, data=evidence)
            predictive = pyro.infer.Predictive(
                model=conditioned,
                num_samples=num_samples,
                return_sites=variables,
                parallel=True,
            )
        else:
            predictive = pyro.infer.Predictive(
                model=self,
                guide=guide,
                num_samples=num_samples,
                return_sites=variables,
                parallel=True,
            )
        raw = predictive(evidence)
        samples = {var: raw[var] for var in variables if var in raw}
        return InferenceOutput(
            samples=samples,
            probs=_concat_sample_means(variables, samples),
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Alias for :class:`BayesianNetwork`.
#: Existing code that instantiates ``ProbabilisticModel(variables, factors)``
#: continues to work and receives a :class:`BayesianNetwork` instance.
ProbabilisticModel = BayesianNetwork
