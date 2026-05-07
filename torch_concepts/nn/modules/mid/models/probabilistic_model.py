"""
:class:`ProbabilisticModel` — owner of variables, factors, and an auto-guide.

See ``mid_level_api.md`` §4.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoNormal

from .variable import Variable, param_dim
from .factor import ParametricFactor
from .cpd import ParametricCPD
from ..constructors.concept_graph import ConceptGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_out_features(module: nn.Module) -> Optional[int]:
    """Best-effort introspection of a parametrization's output dim.

    Returns ``out_features`` of the **last** :class:`nn.Linear` reached
    by ``module.modules()``. ``None`` if no Linear layer is present
    (caller skips validation in that case).
    """
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return None if last is None else last.out_features


def _build_dist(var: Variable, raw: torch.Tensor) -> dist.Distribution:
    """Map a raw parametrization output to a Pyro distribution per spec §2.4."""
    D = var.distribution
    k = var.size
    if D is dist.Bernoulli:
        return dist.Bernoulli(logits=raw).to_event(1)
    if D is dist.Categorical:
        # Categorical: logits shape (..., k=#classes); event_shape=()
        return dist.Categorical(logits=raw)
    if D is dist.Normal:
        loc, raw_scale = raw[..., :k], raw[..., k:]
        scale = torch.nn.functional.softplus(raw_scale) + 1e-6
        return dist.Normal(loc, scale).to_event(1)
    if D is dist.MultivariateNormal:
        loc = raw[..., :k]
        tril_raw = raw[..., k:]
        n_tril = k * (k + 1) // 2
        tril = torch.zeros(*raw.shape[:-1], k, k, device=raw.device, dtype=raw.dtype)
        tri_idx = torch.tril_indices(k, k)
        tril = tril.clone()
        tril[..., tri_idx[0], tri_idx[1]] = tril_raw[..., :n_tril]
        diag = torch.arange(k, device=raw.device)
        # softplus on the diagonal to enforce positive-definiteness
        diag_vals = torch.nn.functional.softplus(tril[..., diag, diag]) + 1e-6
        tril[..., diag, diag] = diag_vals
        return dist.MultivariateNormal(loc, scale_tril=tril)
    if D is dist.Delta:
        return dist.Delta(raw).to_event(1)
    raise ValueError(f"Unsupported distribution {D.__name__}.")


# ---------------------------------------------------------------------------
# ProbabilisticModel
# ---------------------------------------------------------------------------

class ProbabilisticModel(PyroModule):
    """PyroModule owning variables, CPDs, and an auto-guide.

    See spec §4. Behaviour highlights:

    * ``directed=False`` raises :class:`NotImplementedError` (rule §8.7).
    * The auto-guide (default :class:`AutoNormal`) is **eagerly primed**
      with one dummy forward pass so that all variational parameters
      :math:`\\phi` exist before the user constructs an optimizer
      (spec §4.1, design rule §8.1).
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        variables: List[Variable],
        factors: List[ParametricFactor],
        directed: bool = True,
    ):
        super().__init__()

        if not directed:
            raise NotImplementedError(
                "Undirected models are not yet supported. "
                "Pass directed=True or omit this argument."
            )

        self.variables: List[Variable] = list(variables)
        self.concept_to_variable: Dict[str, Variable] = {
            v.concept: v for v in self.variables
        }

        # Sanity: every factor must map to a known variable.
        for f in factors:
            if f.concept not in self.concept_to_variable:
                raise ValueError(
                    f"Factor for concept '{f.concept}' has no matching Variable."
                )

        # Resolve string-named parents to Variable instances.
        for f in factors:
            resolved: List[Variable] = []
            for p in f.parents:
                if isinstance(p, Variable):
                    resolved.append(p)
                elif isinstance(p, str):
                    if p not in self.concept_to_variable:
                        raise ValueError(
                            f"Unknown parent '{p}' for CPD '{f.concept}'."
                        )
                    resolved.append(self.concept_to_variable[p])
                else:
                    raise TypeError(
                        f"Parent must be a Variable or string, got {type(p)}."
                    )
            f.parents = resolved

        # Output-dim validation (spec §8 rule 9). A mismatch raises
        # ``ValueError`` here — semantically equivalent to the spec's
        # "at construction time".
        for f in factors:
            if not isinstance(f, ParametricCPD):
                continue
            if f.parametrization is None:
                continue
            child = self.concept_to_variable[f.concept]
            expected = param_dim(child.distribution, child.size)
            actual = _infer_out_features(f.parametrization)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"Parametrization for CPD '{f.concept}' produces "
                    f"out_features={actual}, but expected "
                    f"param_dim({child.distribution.__name__}, "
                    f"size={child.size}) = {expected}."
                )

        # Register factors as a PyroModule-aware ModuleDict.
        self.factors = PyroModule[nn.ModuleDict]()
        for f in factors:
            self.factors[f.concept] = f

        # Topological sort via ConceptGraph (spec §4.2).
        self.sorted_variables: List[Variable] = self._topological_sort(factors)

        # Eager guide initialization (spec §4.1).
        # AutoNormal lazily creates its loc/scale params on the first call;
        # we run one priming pass so that pgm.parameters() includes phi.
        # Hide discrete sites: AutoNormal can only model continuous latents;
        # discrete ones are tagged for parallel enumeration in `forward`.
        discrete_names = [
            v.concept for v in self.variables
            if getattr(v.distribution, "has_enumerate_support", False)
        ]
        if discrete_names:
            guide_model = poutine.block(self.forward, hide=discrete_names)
        else:
            guide_model = self.forward

        # NOTE: we pass an explicit ``create_plates`` callback so AutoNormal
        # re-creates the data plate with the *current* batch size on every
        # call. Without this, AutoNormal caches the plate size from the
        # priming pass (batch=1) and breaks any subsequent inference whose
        # evidence has a different batch size.
        def _create_plates(evidence, targets=None):
            any_v = next(iter(evidence.values()))
            return [pyro.plate("data", int(any_v.shape[0]), dim=-1)]

        self.guide = AutoNormal(guide_model, create_plates=_create_plates)
        self._prime_guide()

    # ------------------------------------------------------------------
    def _topological_sort(
        self, factors: List[ParametricFactor]
    ) -> List[Variable]:
        names = [v.concept for v in self.variables]
        n = len(names)
        idx = {nm: i for i, nm in enumerate(names)}
        adj = torch.zeros(n, n)
        for f in factors:
            child = idx[f.concept]
            for p in f.parents:
                adj[idx[p.concept], child] = 1.0
        graph = ConceptGraph(adj, node_names=names)
        order = graph.topological_sort()
        return [self.concept_to_variable[name] for name in order]

    def _prime_guide(self) -> None:
        """Run the guide once with synthesized zero evidence.

        Wrapped in :func:`pyro.poutine.block` so that the priming pass does
        **not** write to the global Pyro param store (spec §4.1).
        """
        evidence: Dict[str, torch.Tensor] = {}
        for var in self.variables:
            cpd = self.factors[var.concept]
            if cpd.parametrization is None:
                evidence[var.concept] = torch.zeros((1, var.size))
        if not evidence:
            return  # No roots → cannot synthesize evidence; skip priming.
        with poutine.block():
            self.guide(evidence)

    # ------------------------------------------------------------------
    # Pyro generative model
    # ------------------------------------------------------------------
    def forward(
        self,
        evidence: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Pyro generative model :math:`p(c, z \\mid x)` (spec §4.3).

        Wraps the topological loop in a single mandatory data plate
        (design rule §8.11).
        """
        if not evidence:
            raise ValueError("`evidence` cannot be empty.")
        if targets is None:
            targets = {}

        # Batch size from the leading dim of any evidence tensor.
        any_v = next(iter(evidence.values()))
        batch_size = int(any_v.shape[0])

        values: Dict[str, torch.Tensor] = {}
        with pyro.plate("data", batch_size, dim=-1):
            for var in self.sorted_variables:
                cpd = self.factors[var.concept]

                # Evidence-only root node (spec §3.2 / §8 rule 8).
                if cpd.parametrization is None:
                    if var.concept not in evidence:
                        raise ValueError(
                            f"Root variable '{var.concept}' must be provided "
                            f"in `evidence`."
                        )
                    x = evidence[var.concept]
                    d = dist.Delta(x).to_event(1)
                    values[var.concept] = pyro.sample(var.concept, d, obs=x)
                    continue

                # Non-root: gather realized parents, run parametrization.
                parent_vals = [values[p.concept] for p in cpd.parents]
                if parent_vals:
                    inp = torch.cat(parent_vals, dim=-1)
                else:
                    # Free root with a learnable parametrization (no parents):
                    # 0-feature input with batch dim is the canonical signal.
                    inp = torch.zeros(batch_size, 0)
                raw = cpd.parametrization(inp)
                d = _build_dist(var, raw)

                # Resolve obs from evidence first, then targets, else None.
                obs = evidence.get(var.concept)
                if obs is None:
                    obs = targets.get(var.concept)

                if obs is not None:
                    values[var.concept] = pyro.sample(var.concept, d, obs=obs)
                else:
                    # Discrete-with-finite-support latents are tagged for
                    # parallel enumeration (spec §8 rule 10).
                    if getattr(d, "has_enumerate_support", False):
                        values[var.concept] = pyro.sample(
                            var.concept, d,
                            infer={"enumerate": "parallel"},
                        )
                    else:
                        values[var.concept] = pyro.sample(var.concept, d)


__all__ = ["ProbabilisticModel"]
