"""
ParametricFactor base class for probabilistic graphical models.

This module defines the ParametricFactor base class, which represents a factor
in a factor graph. Factors associate concept-variables with neural network
parametrizations and form the building blocks for both directed (Bayesian
Networks) and undirected (Markov Random Fields) graphical models.
"""
import torch
import torch.nn as nn
from itertools import product
from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical
from typing import Dict, List, Optional, Union

from .variable import Variable


class ParametricFactor(nn.Module):
    """
    Base class for factors in a probabilistic graphical model.

    A ParametricFactor associates a set of named concepts (its *scope*) with a
    single neural-network parametrization.  The factor produces one potential
    over all the concepts in its scope.

    This base class is agnostic to the directionality of the graphical model.
    Subclasses specialise the semantics:

    * :class:`ParametricCPD` — directed factor with explicit ``parents``,
      used in Bayesian Networks.
    * (future) undirected factors for Markov Random Fields.

    Parameters
    ----------
    concepts : Union[str, List[str]]
        A single concept name or a list of concept names defining the scope
        of this factor.  A single ``ParametricFactor`` instance is always
        created (never a list).
    parametrization : nn.Module
        A single neural-network module that computes the factor potential.

    Attributes
    ----------
    concept : str
        Primary concept name (first element of the scope).
    concepts : List[str]
        Full scope of concept names.
    parametrization : nn.Module
        The neural network module used to compute factor values.
    variable : Optional[Variable]
        The :class:`Variable` instance this factor is linked to
        (set by :class:`ProbabilisticModel` during initialisation).

    See Also
    --------
    ParametricCPD : Directed factor for conditional probability distributions.
    Variable : Represents a random variable (concept) in the model.
    ProbabilisticModel : Generic container that manages factors and variables.
    """

    def __init__(self, concepts: Union[str, List[str]],
                 parametrization: nn.Module,
                 **kwargs):
        """
        Initialize a ParametricFactor instance.

        Parameters
        ----------
        concepts : Union[str, List[str]]
            Single concept name (stored as ``self.concept``).
        parametrization : Union[nn.Module, List[nn.Module]]
            Neural network module for computing factor values.
        **kwargs
            Ignored at this level; accepted so that subclass keyword
            arguments (e.g. ``parents``) pass through ``__new__`` without error.
        """
        super().__init__()
        if isinstance(concepts, str):
            self.concepts: List[str] = [concepts]
        else:
            self.concepts: List[str] = list(concepts)
        self.concept: str = self.concepts[0]
        self.parametrization = parametrization
        self.variable: Optional[Variable] = None

    def forward(self, **kwargs):
        """
        Compute the factor output by running the parametrization module.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the parametrization module.

        Returns
        -------
        torch.Tensor
            Output of the parametrization module.
        """
        return self.parametrization(**kwargs)

    # ------------------------------------------------------------------
    # Factor construction for inference algorithms
    # ------------------------------------------------------------------

    def build_factor(
        self,
        variables: List[Variable],
        cardinalities: Dict[str, int] = None,
    ) -> "Factor":
        """
        Build a :class:`Factor` for this undirected potential.

        Enumerates all state combinations of the variables in the
        factor's scope, evaluates the neural-network parametrisation on
        each, and reshapes the result into a multi-dimensional tensor.

        Parameters
        ----------
        variables : List[Variable]
            The :class:`Variable` objects in this factor's scope, in the
            same order as ``self.concepts``.
        cardinalities : dict, optional
            Pre-computed ``{variable_name: num_states}`` mapping.  If
            ``None`` the cardinalities are inferred from *variables*.

        Returns
        -------
        Factor
            A factor whose scope matches ``self.concepts``.
        """
        from .factor import Factor
        from .parametric_cpd import ParametricCPD

        if cardinalities is None:
            cardinalities = {}
        for v in variables:
            cardinalities.setdefault(
                v.concept, ParametricCPD._variable_cardinality(v)
            )

        cards = [cardinalities[c] for c in self.concepts]
        num_combos = 1
        for c in cards:
            num_combos *= c

        # Build the full input batch in a vectorised way.
        # For each variable, construct a column block of shape
        # (num_combos, encoding_dim) using repeat/tile patterns
        # that correspond to itertools.product order.
        blocks: List[torch.Tensor] = []
        repeat_inner = num_combos  # elements that repeat before cycling

        for var, card in zip(variables, cards):
            repeat_inner //= card

            if var.distribution in (Bernoulli, RelaxedBernoulli):
                # States are 0.0 and 1.0 — one feature each.
                states = torch.arange(card, dtype=torch.float32)      # (card,)
                col = states.unsqueeze(1)                              # (card, 1)
            elif var.distribution in (Categorical,
                                      RelaxedOneHotCategorical):
                # One-hot encoding — card features.
                col = torch.eye(card, dtype=torch.float32)             # (card, card)
            else:
                raise TypeError(
                    f"Unsupported distribution "
                    f"{var.distribution.__name__} for factor "
                    f"construction."
                )

            # Tile to match itertools.product order:
            #   each state repeated `repeat_inner` times, whole block
            #   tiled `num_combos // (card * repeat_inner)` times.
            repeat_outer = num_combos // (card * repeat_inner)
            col = col.repeat_interleave(repeat_inner, dim=0)           # (card * repeat_inner, feat)
            col = col.repeat(repeat_outer, 1)                          # (num_combos, feat)
            blocks.append(col)

        input_batch = torch.cat(blocks, dim=-1)                        # (num_combos, total_feat)
        raw = self.parametrization(input=input_batch)  # (num_combos, 1)
        # Squeeze trailing dim if the net outputs a scalar per combo.
        if raw.ndim == 2 and raw.shape[-1] == 1:
            raw = raw.squeeze(-1)
        values = raw.reshape(cards)
        return Factor(values, list(self.concepts), cardinalities)

    def __repr__(self):
        scope = self.concepts if len(self.concepts) > 1 else self.concept
        return f"{self.__class__.__name__}(concepts={scope!r}, parametrization={self.parametrization.__class__.__name__})"
