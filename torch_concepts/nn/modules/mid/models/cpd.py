"""
Conditional Probability Distribution (CPD) for directed probabilistic graphical models.

This module defines the ParametricCPD class, a ParametricFactor subclass that represents
conditional probability distributions in Bayesian Networks.  Unlike the base
ParametricFactor (which only knows about a scope of concepts), a CPD introduces the
notion of **parents** — giving the factor directed semantics.
"""
import copy
import torch
from torch.distributions import Bernoulli, Categorical, OneHotCategorical, RelaxedBernoulli, RelaxedOneHotCategorical
from typing import List, Optional, Tuple, Union
from itertools import product

import torch.nn as nn
from pyro.nn import PyroModule

from .variable import Variable
from .....distributions import Delta


class ParametricCPD(PyroModule):
    """
    Conditional probability distribution parameterised by a neural network.

    Extends :class:`ParametricFactor` with directed-edge semantics: each CPD
    has a list of **parent** concept-variables and computes
    ``P(child | parents)`` via its ``parametrization`` module.

    Parameters
    ----------
    concepts : Union[str, List[str]]
        Concept name(s).  When a list is provided, ``__new__`` returns a list
        of independent ``ParametricCPD`` instances (one per concept).
    parametrization : Union[nn.Module, List[nn.Module]]
        Neural network(s) that compute the conditional distribution.
    parents : List[Union[Variable, str]], optional
        Parent concept-variables (or their names as strings, resolved later
        by :class:`ProbabilisticModel`).  Defaults to ``[]``.

    Attributes
    ----------
    parents : List[Variable]
        Parent concept-variables in the directed graphical model.

    See Also
    --------
    ParametricFactor : Base (undirected) factor class.
    ProbabilisticModel : PGM container that resolves parent references.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __new__(cls,
                concept: Optional[str] = None,
                concepts: Optional[List[str]] = None,
                parametrization: Union[nn.Module, List[nn.Module], dict] = None,
                shared: bool = False,
                **kwargs):
        """Create new ParametricCPD instance(s).

        Exactly one of ``concept`` (single str) or ``concepts`` (list of str)
        must be provided.

        - ``concept`` (str)        → single CPD instance.
        - ``concepts`` (list)      → list of independent CPD instances
          (each with a deep-copied parametrization), unless
          ``shared=True`` in which case a **single** shared CPD is
          returned with its ``parametrization`` shared across all
          concepts (the parametrization must output concatenated logits
          for all concepts).

        ``parametrization`` may also be a **dict** mapping distribution
        parameter names to dedicated ``nn.Module`` instances, e.g.::

            ParametricCPD(concept='z', parametrization={'loc': nn.Linear(8,4),
                                                        'scale': nn.Linear(8,4)})

            # Single module — auto-expanded to one copy per param when the
            # variable's distribution is known (at ProbabilisticModel build time)
            ParametricCPD(concept='z', parametrization=nn.Linear(8,4))
        """
        if concept is not None and concepts is not None:
            raise ValueError(
                "Pass either 'concept' (str) or 'concepts' (List[str]), not both.")
        if concept is None and concepts is None:
            raise ValueError(
                "Must pass either 'concept' (str) or 'concepts' (List[str]).")

        if concept is not None:
            if not isinstance(concept, str):
                raise TypeError(
                    f"'concept' must be a string, got {type(concept).__name__}. "
                    f"Use 'concepts=' for a list of names.")
            if isinstance(parametrization, list):
                raise ValueError(
                    "When 'concept' is provided, 'parametrization' must be a single module, not a list.")
            return object.__new__(cls)

        # concepts is a list
        if not isinstance(concepts, list) or not all(isinstance(c, str) for c in concepts):
            raise TypeError("'concepts' must be a list of strings.")

        # --- shared=True: single instance, no deepcopy ---
        if shared:
            if isinstance(parametrization, list):
                raise ValueError(
                    "When shared=True, 'parametrization' must be a single module, not a list.")
            return object.__new__(cls)

        # --- shared=False (default): one deepcopied instance per concept ---
        n_concepts = len(concepts)

        if isinstance(parametrization, dict):
            instances = []
            for i in range(n_concepts):
                instance = object.__new__(cls)
                instance.__init__(
                    concept=concepts[i],
                    parametrization={k: copy.deepcopy(v) for k, v in parametrization.items()},
                    **kwargs,
                )
                instances.append(instance)
            return instances

        if not isinstance(parametrization, list):
            module_list = [parametrization] * n_concepts
        else:
            module_list = parametrization

        if len(module_list) != n_concepts:
            raise ValueError(
                f"If concepts is a list of length {n_concepts}, parametrization must either be "
                f"a single module or a list of length {n_concepts}.")

        instances = []
        for i in range(n_concepts):
            instance = object.__new__(cls)
            instance.__init__(
                concept=concepts[i],
                parametrization=copy.deepcopy(module_list[i]),
                **kwargs,
            )
            instances.append(instance)
        return instances

    def __init__(self,
                 concept: Optional[str] = None,
                 concepts: Optional[List[str]] = None,
                 parametrization: Union[nn.Module, List[nn.Module], dict] = None,
                 parents: List[Union[Variable, str]] = None,
                 shared: bool = False,
                 shared_name: Optional[str] = None):
        super().__init__()
        if parents is None:
            parents = []

        # Determine primary concept (str) and full concepts list (only set for shared CPDs).
        if concept is not None:
            self.concept = concept
            self.concepts = None
        else:
            # Reached only when shared=True (list path); list path with shared=False
            # creates separate instances per concept via __new__.
            self.concepts = list(concepts)
            self.concept = self.concepts[0]

        # Accept a plain dict and store it as nn.ModuleDict so PyTorch / Pyro
        # correctly register all sub-modules and parameters.
        if isinstance(parametrization, dict) and not isinstance(parametrization, nn.ModuleDict):
            self.parametrization = nn.ModuleDict(parametrization)
        else:
            self.parametrization = parametrization

        assert isinstance(parents, list), "'parents' must be a list of Variable instances or strings."
        self.parents = parents
        self.shared = shared
        self.shared_name = shared_name

        # TODO: use when implementing factors
        # self.scope = concepts+parents if isinstance(concepts, list) else [concepts]+parents

    # ------------------------------------------------------------------
    # Directed-model helpers
    # ------------------------------------------------------------------
    @property
    def in_features(self) -> int:
        """Sum of parent variable sizes."""
        if not self.parents:
            return 0
        return sum(p.size for p in self.parents)

    _MAX_DISCRETE_BITS = 20  # cap on total discrete parent bits for table construction

    # ------------------------------------------------------------------
    # Auto-expansion helper
    # ------------------------------------------------------------------

    def expand_parametrization_to_dict(self) -> None:
        """Auto-expand a single-module parametrization to a per-parameter dict.

        Called by :class:`ProbabilisticModel` after ``self.variable`` has been
        set.  Only expands when the linked variable's distribution has more than
        one parameter group (e.g. ``loc`` + ``scale`` for ``Normal``).
        Creates an independent deep-copy of the original module for each
        distribution parameter so that the parameters can be optimized
        independently.
        """
        if isinstance(self.parametrization, nn.ModuleDict):
            return  # already dict form — nothing to do
        variable = getattr(self, 'variable', None)
        if variable is None:
            return  # no variable linked yet — expansion deferred
        param_keys = list(variable.param_dim.keys())
        if len(param_keys) <= 1:
            return  # single-parameter distribution — no expansion needed
        # Build one deepcopy per distribution parameter
        expanded = nn.ModuleDict({
            k: copy.deepcopy(self.parametrization) for k in param_keys
        })
        self.parametrization = expanded

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _param_forward_order(self):
        """Return the ordered list of param-net keys for the dict form."""
        variable = getattr(self, 'variable', None)
        if variable is not None and isinstance(self.parametrization, nn.ModuleDict):
            # Use _PARAM_DIMS order from the variable
            available = set(self.parametrization.keys())
            ordered = [k for k in variable.param_dim if k in available]
            # Append any extra keys not in _PARAM_DIMS (safety fallback)
            ordered += [k for k in self.parametrization if k not in ordered]
            return ordered
        if isinstance(self.parametrization, nn.ModuleDict):
            return list(self.parametrization.keys())
        return None

    def __repr__(self):
        parents = [p.concept if isinstance(p, Variable) else p for p in self.parents]
        shared = f", shared={self.shared}" if self.shared else ""
        if isinstance(self.parametrization, nn.ModuleDict):
            param_repr = '{' + ', '.join(f'{k}: {v.__class__.__name__}' for k, v in self.parametrization.items()) + '}'
        else:
            param_repr = self.parametrization.__class__.__name__
        if self.concepts is not None:
            id_repr = f"concepts={self.concepts!r}"
        else:
            id_repr = f"concept={self.concept!r}"
        return f"{self.__class__.__name__}({id_repr}, parametrization={param_repr}, parents={parents}{shared})"

    def forward(self, *args, **kwargs):
        """Run the parametrization module(s) and return raw distribution parameters.

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            * If ``parametrization`` is a single ``nn.Module`` → a single
              ``torch.Tensor`` (raw concatenated parameters).
            * If ``parametrization`` is an ``nn.ModuleDict`` (one sub-module
              per distribution parameter, e.g. ``loc`` + ``scale`` for
              ``Normal``) → a ``dict`` mapping each parameter name to its
              own tensor.  This makes the per-parameter dimensions explicit
              instead of relying on positional concatenation.
        """
        if isinstance(self.parametrization, nn.ModuleDict):
            order = self._param_forward_order()
            return {k: self.parametrization[k](*args, **kwargs) for k in order}
        return self.parametrization(*args, **kwargs)

    def sample(self, context: dict, obs=None) -> torch.Tensor:
        """Pyro-aware forward: build distribution from parent context and call ``pyro.sample``.

        Collects parent values from *context*, runs the parametrization to
        produce distribution parameters, builds a Pyro distribution via
        ``variable.make_distribution``, and records a ``pyro.sample`` site.

        Parameters
        ----------
        context : dict
            Maps concept names to already-computed value tensors (the
            running context built by :class:`BayesianNetwork`).
        obs : torch.Tensor, optional
            If provided, the site is marked as observed with this value.

        Returns
        -------
        torch.Tensor
            Sampled (or observed) value for this variable.
        """
        import pyro

        if not self.parents:
            # Root node: input comes through context keyed by concept name.
            raw_input = context.get(self.concept)
            if raw_input is None:
                raise ValueError(
                    f"Root CPD '{self.concept}': no value found in context."
                )
            params = self.forward(raw_input)
        else:
            import inspect
            from .variable import ConceptVariable
            parent_input = []    # latent / exogenous parents
            parent_concepts = [] # concept parents
            for pv in self.parents:
                val = context[pv.concept]
                if isinstance(pv, ConceptVariable):
                    parent_concepts.append(val)
                else:
                    parent_input.append(val)

            # Detect PyC-style signature (has 'concepts' and/or 'latent')
            # For dict-form parametrization inspect the first sub-module.
            try:
                if isinstance(self.parametrization, nn.ModuleDict):
                    first_mod = next(iter(self.parametrization.values()))
                    sig = inspect.signature(first_mod.forward)
                else:
                    sig = inspect.signature(self.parametrization.forward)
                pnames = set(sig.parameters.keys())
            except (ValueError, TypeError):
                pnames = set()

            if 'concepts' in pnames or 'latent' in pnames or 'exogenous' in pnames:
                kw = {}
                if 'concepts' in pnames and parent_concepts:
                    kw['concepts'] = torch.cat(parent_concepts, dim=-1)
                if 'latent' in pnames and parent_input:
                    kw['latent'] = torch.cat(parent_input, dim=-1)
                elif 'exogenous' in pnames and parent_input:
                    kw['exogenous'] = torch.cat(parent_input, dim=-1)
                params = self.forward(**kw)
            else:
                all_vals = parent_concepts + parent_input
                combined = torch.cat(all_vals, dim=-1)
                try:
                    if isinstance(self.parametrization, nn.ModuleDict):
                        first_mod = next(iter(self.parametrization.values()))
                        sig2 = inspect.signature(first_mod.forward)
                    else:
                        sig2 = inspect.signature(self.parametrization.forward)
                    first = next(iter(sig2.parameters))
                    params = self.forward(**{first: combined})
                except (ValueError, TypeError, StopIteration):
                    params = self.forward(combined)

        # Build Pyro distribution and sample
        variable = getattr(self, 'variable', None)
        if variable is None:
            raise RuntimeError(
                f"ParametricCPD '{self.concept}' has no linked Variable. "
                "Ensure it has been registered in a PyroProbabilisticModel."
            )
        d = variable.make_distribution(params)
        return pyro.sample(variable.pyro_site_name, d, obs=obs)
    