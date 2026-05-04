"""
PyroAmortizedGuide — auto-built amortized variational guide for CBMs.

The guide reuses the same :class:`ParametricCPD` parametrization networks
that are present in the :class:`PyroProbabilisticModel`.  For every
non-Delta variable that is **not** observed in the provided *data* dict,
the guide samples from the distribution parameterised by the CPD encoder.

This amounts to the "recognition network = generative network" choice,
which is appropriate for supervised CBMs where concepts can be directly
supervised.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

import pyro
from pyro.nn import PyroModule

from ..models.variable import Variable, ConceptVariable
from ..models.cpd import ParametricCPD


class AmortizedGuide(PyroModule):
    """
    Amortized variational guide auto-built from a
    :class:`PyroProbabilisticModel`'s CPD networks.

    For each non-Delta variable that is not observed at runtime (i.e. not
    present in *data*), this guide:

    1. Collects parent values from the running context.
    2. Runs the same ``ParametricCPD.parametrization`` as the model.
    3. Calls ``pyro.sample(name, dist)`` (without ``obs``).

    Parameters
    ----------
    pgm : PyroProbabilisticModel
        The Pyro-backed PGM whose CPD networks are reused.
    """

    def __init__(self, pgm: 'PyroProbabilisticModel') -> None:  # noqa: F821
        super().__init__()
        # Store the PGM as a plain Python attribute (bypassing PyroModule's
        # __setattr__) so Pyro does not try to register it as a child
        # PyroModule — that would trigger an assertion error if pgm has
        # already been executed in a Pyro context.
        object.__setattr__(self, '_pgm', pgm)
        # Register the CPD parametrization networks (plain nn.Module objects)
        # directly on the guide.  Because we use the *same* nn.Module instances
        # as the model, weights are shared: gradients and parameter updates are
        # the same regardless of whether the forward pass goes through the model
        # or the guide.  Storing them here also makes them appear in
        # guide.parameters() so PyroInferenceEngine can collect them.
        import torch.nn as nn
        self.encoders = nn.ModuleDict({
            name: cpd.parametrization
            for name, cpd in pgm.factors.items()
        })

    def forward(
        self,
        evidence: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Guide forward pass — sample all latent (non-observed) variables.

        Parameters
        ----------
        evidence : dict
            Same evidence dict passed to the model.  Variables present here
            (or in *targets*) are **not** re-sampled by the guide — they are
            already conditioned on in the model.
        targets : dict, optional
            Additional observed tensors (e.g. concept labels).  Merged with
            *evidence* to determine which variables are observed.

        Returns
        -------
        dict
            Running context mapping concept names to tensors.
        """
        from ..models.probabilistic_model import ProbabilisticModel as _PyroModel

        # Merge evidence and targets so the guide sees all observed values.
        obs_dict: Dict[str, torch.Tensor] = (
            {**evidence, **targets} if targets else evidence
        )

        pgm = self._pgm
        batch_size = next(iter(obs_dict.values())).shape[0]
        context: Dict[str, torch.Tensor] = {}

        with pyro.plate('data', batch_size):
            for var in pgm.sorted_variables:
                name = var.concept
                cpd = pgm.get_module_of_concept(name)
                if cpd is None:
                    continue

                if var.is_deterministic:
                    # Deterministic: propagate without sampling
                    if not cpd.parents:
                        key = getattr(cpd, 'shared_name', None) or name
                        val = obs_dict.get(key)
                        if val is None:
                            val = obs_dict.get(name)
                        context[name] = val
                    else:
                        kwargs = _PyroModel._build_parent_kwargs(cpd, context)
                        context[name] = cpd.parametrization(**kwargs)

                elif name in obs_dict:
                    # Observed: guide does not sample; just track value
                    context[name] = obs_dict[name]

                else:
                    # Latent / hidden: guide samples
                    if not cpd.parents:
                        key = getattr(cpd, 'shared_name', None) or name
                        raw = obs_dict.get(key)
                        if raw is None:
                            raw = obs_dict.get(name)
                        if raw is None:
                            raise ValueError(
                                f"Guide: root variable '{name}' not in evidence."
                            )
                        params = cpd.parametrization(raw)
                    else:
                        kwargs = _PyroModel._build_parent_kwargs(cpd, context)
                        params = cpd.parametrization(**kwargs)

                    d = var.make_distribution(params)
                    context[name] = pyro.sample(name, d)

        return context
