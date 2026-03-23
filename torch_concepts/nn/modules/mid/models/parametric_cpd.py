"""
Conditional Probability Distribution (CPD) for directed probabilistic graphical models.

This module defines the ParametricCPD class, a ParametricFactor subclass that represents
conditional probability distributions in Bayesian Networks.  Unlike the base
ParametricFactor (which only knows about a scope of concepts), a CPD introduces the
notion of **parents** — giving the factor directed semantics.
"""
import copy
import torch
from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical
from typing import List, Optional, Tuple, Union
from itertools import product

import torch.nn as nn

from .parametric_factor import ParametricFactor
from .factor import Factor
from .variable import Variable
from .....distributions import Delta


class ParametricCPD(ParametricFactor):
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
    def __new__(cls, concepts: Union[str, List[str]],
                parametrization: Union[nn.Module, List[nn.Module]],
                **kwargs):
        """
        Create new ParametricCPD instance(s).

        If ``concepts`` is a string, returns a single instance.
        If ``concepts`` is a list, returns a list of instances (one per concept),
        each with a deep-copied parametrization.
        """
        if isinstance(concepts, str):
            if isinstance(parametrization, list):
                raise ValueError(
                    "When 'concepts' is a string, 'parametrization' must be a single module, not a list.")
            return object.__new__(cls)

        n_concepts = len(concepts)
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
                concepts=concepts[i],
                parametrization=copy.deepcopy(module_list[i]),
                **kwargs,
            )
            instances.append(instance)
        return instances

    def __init__(self, concepts: Union[str, List[str]],
                 parametrization: Union[nn.Module, List[nn.Module]],
                 parents: List[Union[Variable, str]] = None,
                 **kwargs):
        super().__init__(concepts=concepts, parametrization=parametrization, **kwargs)
        self.parents: List[Variable] = list(parents) if parents is not None else []

    # ------------------------------------------------------------------
    # Directed-model helpers (moved from ParametricFactor)
    # ------------------------------------------------------------------
    @property
    def in_features(self) -> int:
        """Sum of parent variable sizes."""
        if not self.parents:
            return 0
        return sum(p.size for p in self.parents)

    def _get_parent_combinations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enumerate all discrete parent-state combinations for table construction.

        Continuous (Delta / Normal) parents are held at zero; discrete parents
        (Bernoulli, Categorical and their relaxed variants) are exhaustively
        enumerated.

        Returns
        -------
        all_full_inputs : torch.Tensor
            Input tensors for the parametrization, one row per combination.
        all_discrete_state_vectors : torch.Tensor
            Corresponding state vectors for the table rows.
        """
        if not self.parents:
            in_features = self.parametrization.in_features
            placeholder_input = torch.zeros((1, in_features))
            return placeholder_input, torch.empty((1, 0))

        discrete_combinations_list = []
        discrete_state_vectors_list = []
        continuous_tensors = []

        for parent_var in self.parents:
            if parent_var.distribution in [Bernoulli, RelaxedBernoulli,
                                           Categorical, RelaxedOneHotCategorical]:
                out_dim = parent_var.out_features
                input_combinations = []
                state_combinations = []

                if parent_var.distribution in [Bernoulli, RelaxedBernoulli]:
                    input_combinations = list(product([0.0, 1.0], repeat=out_dim))
                    state_combinations = input_combinations
                elif parent_var.distribution in [Categorical, RelaxedOneHotCategorical]:
                    for i in range(out_dim):
                        one_hot = torch.zeros(out_dim)
                        one_hot[i] = 1.0
                        input_combinations.append(one_hot.tolist())
                        state_combinations.append([float(i)])

                discrete_combinations_list.append(
                    [torch.tensor(c, dtype=torch.float32).unsqueeze(0) for c in input_combinations])
                discrete_state_vectors_list.append(
                    [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in state_combinations])

            elif parent_var.distribution is Delta or parent_var.distribution is torch.distributions.Normal:
                fixed_value = torch.zeros(parent_var.out_features).unsqueeze(0)
                continuous_tensors.append(fixed_value)
            else:
                raise TypeError(
                    f"Unsupported distribution type {parent_var.distribution.__name__} for table generation.")

        if not discrete_combinations_list:
            fixed_continuous_input = (torch.cat(continuous_tensors, dim=-1)
                                      if continuous_tensors else torch.empty((1, 0)))
            return fixed_continuous_input, torch.empty((1, 0))

        all_discrete_product = list(product(*discrete_combinations_list))
        all_discrete_states_product = list(product(*discrete_state_vectors_list))

        fixed_continuous_input = (torch.cat(continuous_tensors, dim=-1)
                                  if continuous_tensors else torch.empty((1, 0)))

        all_full_inputs = []
        for discrete_inputs in all_discrete_product:
            discrete_part = torch.cat(list(discrete_inputs), dim=-1)
            all_full_inputs.append(torch.cat([discrete_part, fixed_continuous_input], dim=-1))

        all_discrete_state_vectors = []
        for discrete_states in all_discrete_states_product:
            all_discrete_state_vectors.append(torch.cat(list(discrete_states), dim=-1))

        return torch.cat(all_full_inputs, dim=0), torch.cat(all_discrete_state_vectors, dim=0)

    # ------------------------------------------------------------------
    # CPT / potential-table construction
    # ------------------------------------------------------------------

    def build_cpt(self) -> torch.Tensor:
        if not self.variable:
            raise RuntimeError("ParametricCPD not linked to a Variable in ProbabilisticModel.")

        all_full_inputs, discrete_state_vectors = self._get_parent_combinations()

        input_batch = all_full_inputs

        if input_batch.shape[-1] != self.parametrization.in_features:
            raise RuntimeError(
                f"Input tensor dimension mismatch for CPT building. "
                f"ParametricCPD module expects {self.parametrization.in_features} features, "
                f"but parent combinations resulted in {input_batch.shape[-1]} features. "
                f"Check Variable definition and ProbabilisticModel resolution."
            )

        endogenous = self.parametrization(input=input_batch)
        probabilities = None

        if self.variable.distribution is Bernoulli:
            # Traditional P(X=1) output
            p_c1 = torch.sigmoid(endogenous)

            # ACHIEVE THE REQUESTED 4x3 STRUCTURE: [Parent States | P(X=1)]
            probabilities = torch.cat([discrete_state_vectors, p_c1], dim=-1)

        elif self.variable.distribution is Categorical:
            probabilities = torch.softmax(endogenous, dim=-1)

        elif self.variable.distribution is Delta:
            probabilities = endogenous

        else:
            raise NotImplementedError(f"CPT for {self.variable.distribution.__name__} not supported.")

        return probabilities

    def build_potential(self) -> torch.Tensor:
        if not self.variable:
            raise RuntimeError("ParametricCPD not linked to a Variable in ProbabilisticModel.")

        # We need the core probability part for potential calculation
        all_full_inputs, discrete_state_vectors = self._get_parent_combinations()
        endogenous = self.parametrization(input=all_full_inputs)

        if self.variable.distribution is Bernoulli:
            cpt_core = torch.sigmoid(endogenous)
        elif self.variable.distribution is Categorical:
            cpt_core = torch.softmax(endogenous, dim=-1)
        elif self.variable.distribution is Delta:
            cpt_core = endogenous
        else:
            raise NotImplementedError("Potential table construction not supported for this distribution.")

        # --- Potential Table Construction ---

        if self.variable.distribution is Bernoulli:
            p_c1 = cpt_core
            p_c0 = 1.0 - cpt_core

            child_states_c0 = torch.zeros_like(p_c0)
            child_states_c1 = torch.ones_like(p_c1)

            # Rows for X=1: [Parent States | Child State (1) | P(X=1)]
            rows_c1 = torch.cat([discrete_state_vectors, child_states_c1, p_c1], dim=-1)
            # Rows for X=0: [Parent States | Child State (0) | P(X=0)]
            rows_c0 = torch.cat([discrete_state_vectors, child_states_c0, p_c0], dim=-1)

            potential_table = torch.cat([rows_c1, rows_c0], dim=0)

        elif self.variable.distribution is Categorical:
            n_classes = self.variable.size
            all_rows = []
            for i in range(n_classes):
                child_state_col = torch.full((cpt_core.shape[0], 1), float(i), dtype=torch.float32)
                prob_col = cpt_core[:, i].unsqueeze(-1)

                # [Parent States | Child State (i) | P(X=i)]
                rows_ci = torch.cat([discrete_state_vectors, child_state_col, prob_col], dim=-1)
                all_rows.append(rows_ci)

            potential_table = torch.cat(all_rows, dim=0)

        elif self.variable.distribution is Delta:
            # [Parent States | Child Value]
            child_value = cpt_core
            potential_table = torch.cat([discrete_state_vectors, child_value], dim=-1)

        else:
            raise NotImplementedError("Potential table construction not supported for this distribution.")

        return potential_table

    # ------------------------------------------------------------------
    # Factor construction for inference algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def _variable_cardinality(var: Variable) -> int:
        """Return the number of discrete states for a variable."""
        if var.distribution in (Bernoulli, RelaxedBernoulli):
            return 2
        elif var.distribution in (Categorical, RelaxedOneHotCategorical):
            return var.size
        else:
            raise TypeError(
                f"Cannot compute cardinality for distribution "
                f"{var.distribution.__name__}. Only discrete distributions "
                f"(Bernoulli, Categorical) are supported for factor "
                f"construction."
            )

    def build_factor(self, cardinalities: dict = None) -> "Factor":
        """
        Build a :class:`Factor` from this CPD.

        Evaluates the neural-network parametrisation over all discrete
        parent-state combinations and reshapes the result into a
        multi-dimensional tensor whose axes are
        ``[parent_1, parent_2, …, child]``.

        For a Bayesian Network, each entry stores
        ``P(child = c | parents = pa)``; the Factor is therefore the
        *conditional* distribution viewed as a joint potential (un-normalised
        over the full scope).

        Parameters
        ----------
        cardinalities : dict, optional
            Pre-computed ``{variable_name: num_states}`` mapping.  If
            ``None`` the cardinalities are inferred from the parent and
            child :class:`Variable` objects.

        Returns
        -------
        Factor
            A factor whose scope is ``[parent_1, …, parent_n, child]``
            and whose values are the conditional probabilities produced
            by the neural network.

        Raises
        ------
        RuntimeError
            If the CPD has not been linked to a Variable.
        TypeError
            If any parent or the child has a non-discrete distribution.
        """
        if not self.variable:
            raise RuntimeError(
                "ParametricCPD not linked to a Variable in "
                "ProbabilisticModel."
            )

        # --- determine scope and cardinalities ---
        # Only consider discrete parents (continuous parents are held
        # fixed at zero — same convention as _get_parent_combinations).
        discrete_parents = [
            p for p in self.parents
            if p.distribution in (Bernoulli, RelaxedBernoulli,
                                  Categorical, RelaxedOneHotCategorical)
        ]

        scope: List[str] = [p.concept for p in discrete_parents]
        scope.append(self.variable.concept)

        if cardinalities is None:
            cardinalities = {}
        # Fill in cardinalities from the variables themselves.
        for p in discrete_parents:
            cardinalities.setdefault(p.concept, self._variable_cardinality(p))
        cardinalities.setdefault(
            self.variable.concept,
            self._variable_cardinality(self.variable),
        )

        child_card = cardinalities[self.variable.concept]
        parent_cards = [cardinalities[p.concept] for p in discrete_parents]

        # --- evaluate the neural network ---
        all_full_inputs, _ = self._get_parent_combinations()
        endogenous = self.parametrization(input=all_full_inputs)

        if self.variable.distribution in (Bernoulli, RelaxedBernoulli):
            p1 = torch.sigmoid(endogenous)          # (num_combos, 1)
            probs = torch.cat([1.0 - p1, p1], dim=-1)  # (num_combos, 2)
        elif self.variable.distribution in (Categorical,
                                            RelaxedOneHotCategorical):
            probs = torch.softmax(endogenous, dim=-1)   # (num_combos, K)
        else:
            raise TypeError(
                f"build_factor() requires a discrete child distribution, "
                f"got {self.variable.distribution.__name__}."
            )

        # --- reshape into multi-dimensional tensor ---
        # `_get_parent_combinations` enumerates in row-major
        # (itertools.product) order over the discrete parents.
        # probs shape: (prod(parent_cards), child_card)
        tensor_shape = parent_cards + [child_card]
        values = probs.reshape(tensor_shape)

        return Factor(values, scope, cardinalities)
