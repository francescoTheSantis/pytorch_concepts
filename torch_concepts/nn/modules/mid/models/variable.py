"""
Variable representation for concept-based Probabilistic Models.

This module defines the Variable class, which represents random variables in
concept-based models. Variables can have different probability distributions
and support hierarchical concept structures.
"""
import copy
import torch
from functools import partial

# PyTorch distributions
from torch.distributions import Distribution, Normal, MultivariateNormal, \
    Bernoulli, RelaxedBernoulli, Categorical, OneHotCategorical, RelaxedOneHotCategorical
from .....distributions import Delta

from typing import List, Dict, Any, Union, Optional, Type, Callable


# List of supported distribution types for Variables.
_SUPPORTED_DISTRIBUTIONS: list = [
    Bernoulli,
    RelaxedBernoulli,
    OneHotCategorical,
    RelaxedOneHotCategorical,
    Normal,
    MultivariateNormal,
    Delta
]

# Default size for distributions that have a natural scalar default.
_DEFAULT_SIZES: Dict[Type[Distribution], int] = {
    Bernoulli: 1,
    RelaxedBernoulli: 1,
    Normal: 1
}

# Default distributions per concept type group (binary / categorical / continuous).
_DEFAULT_DISTRIBUTIONS: Dict[str, Type[Distribution]] = {
    'binary': RelaxedBernoulli,
    'categorical': RelaxedOneHotCategorical,
    'continuous': Normal
}

# Default dist_kwargs for distributions that require constructor arguments.
_DEFAULT_DIST_KWARGS: Dict[Type[Distribution], Dict[str, Any]] = {
    RelaxedBernoulli: {'temperature': 0.5},
    RelaxedOneHotCategorical: {'temperature': 0.5},
}

# Number of raw parameters needed to parameterise each supported distribution
# given a variable of a certain *size* (event dimension).
_PARAM_DIMS: Dict[Type[Distribution], Dict[str, Callable[[int], int]]] = {
    Bernoulli: {'logits': lambda size: size},                        # one logit per dimension
    RelaxedBernoulli: {'logits': lambda size: size},                 # one logit per dimension
    OneHotCategorical: {'logits': lambda size: size},                # one logit per class
    RelaxedOneHotCategorical: {'logits': lambda size: size},         # one logit per class
    Normal: {'loc': lambda size: size,                               # mean
             'scale': lambda size: size},                            # log-std
    MultivariateNormal: {'loc': lambda size: size,                            # mean
                         'scale_tril': lambda size: size * (size + 1) // 2},  # lower-triangular Cholesky
    Delta: {'value': lambda size: size}
}


def param_dim(
    distribution: Type[Distribution], 
    size: int = 1,
    return_sum: bool = True) -> int:
    """Return the number of raw parameters required to parameterise *distribution*.

    Args:
        distribution: A supported distribution class (must be in
            :data:`_SUPPORTED_DISTRIBUTIONS`).
        size: The event dimension of the variable (e.g. ``1`` for
            Bernoulli, ``k`` for ``OneHotCategorical`` with *k* classes,
            ``d`` for a ``d``-dimensional ``Normal`` or
            ``MultivariateNormal``).

    Returns:
        int: Total number of scalar parameters.

    Raises:
        ValueError: If *distribution* is not in :data:`_SUPPORTED_DISTRIBUTIONS`.
    """
    if distribution not in _SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"Distribution '{distribution.__name__}' is not supported. "
            f"Supported distributions are: {[d.__name__ for d in _SUPPORTED_DISTRIBUTIONS]}."
        )
    if return_sum:
        return sum(v(size) for v in _PARAM_DIMS[distribution].values()) 
    else:        
        return {k: v(size) for k, v in _PARAM_DIMS[distribution].items()}


class Variable:
    """
    Represents a random variable in a concept-based Probabilistic Model.

    A Variable encapsulates one or more concepts along with their associated
    probability distribution and metadata. It supports multiple distribution
    types including Delta (deterministic), Bernoulli, Categorical, and Normal
    distributions.

    The Variable class implements a special __new__ method that allows creating
    multiple Variable instances when initialized with multiple concepts, or a
    single instance for a single concept.

    Attributes:
        concept (str): The concept name represented by this variable.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Size/cardinality of the variable (e.g., number of classes for Categorical).
        dist_kwargs (Dict[str, Any]): Keyword arguments passed to the distribution constructor
            (e.g., ``{'temperature': 0.5}`` for relaxed distributions).
        metadata (Dict[str, Any]): Additional metadata associated with the variable.

    Properties:
        out_features (int): Number of output features this variable produces.

    Example:
        >>> import torch
        >>> from torch.distributions import Bernoulli, Categorical, Normal
        >>> from torch_concepts import Variable
        >>> from torch_concepts.distributions import Delta
        >>>
        >>> # Create a binary concept variable
        >>> var_binary = Variable(
        ...     concepts='has_wheels',
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> print(var_binary.concept)  # 'has_wheels'
        >>> print(var_binary.out_features)  # 1
        >>>
        >>> # Create a categorical variable with 3 color classes
        >>> var_color = Variable(
        ...     concepts=['color'],
        ...     distribution=OneHotCategorical,
        ...     size=3  # red, green, blue
        ... )
        >>> print(var_color.out_features)  # 3
        >>>
        >>> # Create multiple variables at once
        >>> vars_list = Variable(
        ...     concepts=['A', 'B', 'C'],
        ...     distribution=Delta,
        ...     size=1
        ... )
        >>> print(len(vars_list))  # 3
        >>> print(vars_list[0].concept)  # 'A'
        >>> print(vars_list[1].concept)  # 'B'
    """

    def __new__(cls, concept: Optional[str] = None,
                concepts: Optional[List[str]] = None,
                distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                size: Union[int, List[int], None] = None,
                metadata: Optional[Dict[str, Any]] = None,
                dist_kwargs: Optional[Dict[str, Any]] = None,
                observed: Optional[bool] = None,
                **kwargs):
        """Create new Variable instance(s).

        Exactly one of ``concept`` (single str) or ``concepts`` (list of str)
        must be provided.  When ``concept`` is given, returns a single
        Variable instance.  When ``concepts`` is given, returns a list of
        Variable instances (one per name) sharing the same other arguments.
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
            if isinstance(distribution, list):
                raise ValueError(
                    "When 'concept' is provided, 'distribution' must be a single value, not a list.")
            if isinstance(size, list):
                raise ValueError(
                    "When 'concept' is provided, 'size' must be a single value, not a list.")
            return object.__new__(cls)

        # concepts is a non-empty list -> return list of Variables
        if not isinstance(concepts, list) or not all(isinstance(c, str) for c in concepts):
            raise TypeError("'concepts' must be a list of strings.")

        n_concepts = len(concepts)

        # Standardize distribution: single value -> list of N values
        if distribution is None:
            distribution_list = [Delta] * n_concepts
        elif not isinstance(distribution, list):
            distribution_list = [distribution] * n_concepts
        else:
            distribution_list = distribution

        # Standardize size: single value -> list of N values
        if not isinstance(size, list):
            size_list = [size] * n_concepts
        else:
            size_list = size

        if len(distribution_list) != n_concepts or len(size_list) != n_concepts:
            raise ValueError(
                f"If concepts is a list of length {n_concepts}, distribution and size must either be "
                f"single values or lists of length {n_concepts}.")

        new_vars = []
        for i in range(n_concepts):
            instance = object.__new__(cls)
            instance.__init__(
                concept=concepts[i],
                distribution=distribution_list[i],
                size=size_list[i],
                metadata=copy.deepcopy(metadata) if metadata else None,
                dist_kwargs=copy.deepcopy(dist_kwargs) if dist_kwargs else None,
                observed=observed,
                **kwargs,
            )
            new_vars.append(instance)
        return new_vars

    def __init__(self, concept: Optional[str] = None,
                 concepts: Optional[List[str]] = None,
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int], None] = None,
                 metadata: Dict[str, Any] = None,
                 dist_kwargs: Optional[Dict[str, Any]] = None,
                 observed: Optional[bool] = None,
                 **kwargs):
        """Initialize a single Variable instance.

        Always called with ``concept`` (str). The list path goes through
        ``__new__`` which dispatches to per-concept ``__init__`` calls.
        """
        # Single-instance path: ``concept`` carries the name.
        # Original validation logic
        if distribution is None:
            distribution = Delta

        if distribution is Categorical:
            raise ValueError(
                "Categorical.sample() returns a class index, not a one-hot vector, "
                "which is incompatible with Variable. "
                "Use OneHotCategorical (or RelaxedOneHotCategorical) instead."
            )
        
        if distribution not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Distribution '{distribution.__name__}' is not supported. "
                f"Supported distributions are: {[d.__name__ for d in _SUPPORTED_DISTRIBUTIONS]}."
            )

        if size is None:
            if distribution in _DEFAULT_SIZES:
                size = _DEFAULT_SIZES[distribution]
            else:
                raise ValueError(
                    f"'size' must be provided for distribution '{distribution.__name__}'."
                )

        if distribution in [Bernoulli, RelaxedBernoulli] and size != 1:
            raise ValueError("Bernoulli and RelaxedBernoulli distributions must have size=1.")

        if distribution is MultivariateNormal and size < 2:
            raise ValueError(
                "MultivariateNormal requires size >= 2 (the event dimension). "
                "For a 1-D normal distribution use Normal instead."
            )

        self.concept = concept
        self.distribution = distribution
        self.size = size
        self.dist_kwargs = dist_kwargs if dist_kwargs is not None else {}
        self.metadata = metadata if metadata is not None else {}
        self._observed = observed  # explicit override; None means use type-based default

    @property
    def out_features(self) -> int:
        """
        Number of output features for this variable.

        Returns:
            int: Number of output features.
        """
        return param_dim(self.distribution, self.size, return_sum=True)

    @property
    def param_dim(self) -> int:
        """
        Number of output features for this variable.

        Returns:
            int: Number of output features.
        """
        return param_dim(self.distribution, self.size, return_sum=False)

    @property
    def pyro_site_name(self) -> str:
        """Pyro sample-site name (equals concept name)."""
        return self.concept

    @property
    def is_observed(self) -> bool:
        """Whether this variable is observed (present in the evidence dict).

        Defaults to ``True`` for :class:`ExogenousVariable` and ``False`` for
        all other types, but can be overridden per-instance via the
        ``observed`` constructor parameter.
        """
        if self._observed is not None:
            return self._observed
        return self.metadata.get('variable_type') == 'exogenous'

    @property
    def is_deterministic(self) -> bool:
        """True if this variable is deterministic (uses the Delta distribution).

        Deterministic variables are propagated via ``pyro.deterministic``
        rather than sampled.  Use ``distribution=Delta`` to create a
        deterministic variable.
        """
        from .....distributions import Delta as _Delta
        return self.distribution is _Delta

    def make_distribution(self, params) -> 'pyro.distributions.Distribution':
        """Build a Pyro distribution from CPD output *params*.

        ``params`` may be either a flat ``torch.Tensor`` (single-parameter
        distributions, or multi-parameter distributions whose parameters
        have been concatenated along the last dim) **or** a ``dict`` mapping
        distribution parameter names to per-parameter tensors (the form
        produced by a multi-parameter :class:`ParametricCPD`).

        For multi-parameter distributions the dict form is preferred because
        it makes the per-parameter dimensions explicit.
        """
        import pyro.distributions as pyd
        import torch.nn.functional as F

        kwargs = self.dist_kwargs
        is_dict = isinstance(params, dict)

        if self.distribution is Delta:
            value = params['value'] if is_dict else params
            event_dim = max(0, value.dim() - 1)
            return pyd.Delta(value, event_dim=event_dim)

        if self.distribution is Bernoulli:
            logits = params['logits'] if is_dict else params
            return pyd.Bernoulli(logits=logits).to_event(1)

        if self.distribution is RelaxedBernoulli:
            temperature = kwargs.get('temperature', 0.5)
            logits = params['logits'] if is_dict else params
            t = torch.tensor(temperature, dtype=logits.dtype, device=logits.device)
            return pyd.RelaxedBernoulli(temperature=t, logits=logits).to_event(1)

        if self.distribution is OneHotCategorical:
            logits = params['logits'] if is_dict else params
            return pyd.OneHotCategorical(logits=logits)

        if self.distribution is RelaxedOneHotCategorical:
            temperature = kwargs.get('temperature', 0.5)
            logits = params['logits'] if is_dict else params
            t = torch.tensor(temperature, dtype=logits.dtype, device=logits.device)
            return pyd.RelaxedOneHotCategorical(temperature=t, logits=logits)

        if self.distribution is Normal:
            if is_dict:
                loc = params['loc']
                scale = F.softplus(params['scale']) + 1e-6
            else:
                size = self.size
                loc = params[..., :size]
                scale = F.softplus(params[..., size:]) + 1e-6
            return pyd.Normal(loc, scale).to_event(1)

        if self.distribution is MultivariateNormal:
            d = self.size
            if is_dict:
                loc = params['loc']
                tril_flat = params['scale_tril']
            else:
                loc = params[..., :d]
                tril_flat = params[..., d:]  # (..., d*(d+1)//2)
            batch_shape = loc.shape[:-1]
            scale_tril = torch.zeros(*batch_shape, d, d,
                                     device=loc.device, dtype=loc.dtype)
            rows, cols = torch.tril_indices(d, d, device=loc.device)
            scale_tril[..., rows, cols] = tril_flat
            # Ensure positive diagonal via softplus
            diag_idx = torch.arange(d, device=loc.device)
            scale_tril[..., diag_idx, diag_idx] = (
                F.softplus(scale_tril[..., diag_idx, diag_idx]) + 1e-6
            )
            return pyd.MultivariateNormal(loc, scale_tril=scale_tril)

        raise ValueError(
            f"make_distribution: distribution '{self.distribution.__name__}' is not supported."
        )

    def __repr__(self):
        """
        Return string representation of the Variable.

        Returns:
            str: String representation including concepts, distribution, size, and metadata.
        """
        meta_str = f"metadata={self.metadata}" if self.metadata else ""
        dist_kwargs_str = f"({self.dist_kwargs})" if self.dist_kwargs else ""
        return f"Variable(concept='{self.concept}', dist={self.distribution.__name__}{dist_kwargs_str}, size={self.size}, param_dim={self.param_dim}, {meta_str})"


class ConceptVariable(Variable):
    """
    Represents a concept variable in a concept-based model.
    
    Concept variables are observable and supervisable variables that can be
    directly measured or annotated in the data. These are typically the concepts
    that we want to learn and predict, such as object attributes, semantic features,
    or intermediate representations that have ground truth labels.
    
    Attributes:
        concept (str): The concept name represented by this variable.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Size/cardinality of the variable.
        dist_kwargs (Dict[str, Any]): Keyword arguments for the distribution constructor.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'concept'.
        
    Example:
        >>> from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli
        >>> from torch_concepts import ConceptVariable
        >>> # Observable binary concept
        >>> has_wings = ConceptVariable(
        ...     concepts='has_wings',
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> 
        >>> # Relaxed binary concept with temperature
        >>> has_wings_relaxed = ConceptVariable(
        ...     concepts='has_wings',
        ...     distribution=RelaxedBernoulli,
        ...     size=1,
        ...     dist_kwargs={'temperature': 0.5}
        ... )
        >>> 
        >>> # Observable categorical concept (e.g., color)
        >>> color = ConceptVariable(
        ...     concepts=['color'],
        ...     distribution=OneHotCategorical,
        ...     size=3  # red, green, blue
        ... )
    """
    
    def __init__(self, concept: Optional[str] = None,
                 concepts: Optional[List[str]] = None,
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int]] = 1,
                 metadata: Dict[str, Any] = None,
                 dist_kwargs: Optional[Dict[str, Any]] = None,
                 observed: bool = False,
                 **kwargs):
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'concept'
        super().__init__(concept=concept, concepts=concepts,
                         distribution=distribution, size=size,
                         metadata=metadata, dist_kwargs=dist_kwargs,
                         observed=observed, **kwargs)


# Backward compatibility alias
EndogenousVariable = ConceptVariable


class ExogenousVariable(Variable):
    """
    Represents an exogenous variable in a concept-based model.
    
    Exogenous variables are high-dimensional representations related to a single
    concept variable. They capture rich, detailed information about a specific
    concept (e.g., image patches, embeddings, or feature vectors) that can be used
    to predict or explain the corresponding concept.
    
    Attributes:
        concept (str): The concept name represented by this variable.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Dimensionality of the high-dimensional representation.
        concept_var (Optional[ConceptVariable]): The concept variable this exogenous variable is related to.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'exogenous'.
        
    Example:
        >>> from torch.distributions import Normal, Bernoulli
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts import ConceptVariable, ExogenousVariable
        >>> # Concept variable
        >>> has_wings = ConceptVariable(
        ...     concepts='has_wings',
        ...     distribution=Bernoulli,
        ...     size=1
        ... )
        >>> 
        >>> # Exogenous high-dim representation for has_wings
        >>> wings_features = ExogenousVariable(
        ...     concepts='wings_exogenous',
        ...     distribution=Delta,
        ...     size=128,  # 128-dimensional exogenous
        ... )
    """
    
    def __init__(self, concept: Optional[str] = None,
                 concepts: Optional[List[str]] = None,
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int]] = 1,
                 concept_var: Optional['ConceptVariable'] = None,
                 metadata: Dict[str, Any] = None,
                 dist_kwargs: Optional[Dict[str, Any]] = None,
                 observed: bool = True,
                 **kwargs):
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'exogenous'
        if concept_var is not None:
            metadata['concept_var'] = concept_var
        super().__init__(concept=concept, concepts=concepts,
                         distribution=distribution, size=size,
                         metadata=metadata, dist_kwargs=dist_kwargs,
                         observed=observed, **kwargs)
        self.concept_var = concept_var


class LatentVariable(Variable):
    """
    Represents a latent variable in a concept-based model.
    
    Latent variables are high-dimensional global representations of the whole input
    object (e.g., raw input images, text, or sensor data). They capture the complete
    information about the input before it is decomposed into specific concepts.
    These are typically unobserved, learned representations that encode all relevant
    information from the raw input.
    
    Attributes:
        concept (str): The concept name represented by this variable.
        distribution (Type[Distribution]): PyTorch distribution class for this variable.
        size (int): Dimensionality of the latent representation.
        dist_kwargs (Dict[str, Any]): Keyword arguments for the distribution constructor.
        metadata (Dict[str, Any]): Additional metadata. Automatically includes 'variable_type': 'latent'.
        
    Example:
        >>> from torch_concepts.distributions import Delta
        >>> from torch_concepts import LatentVariable
        >>> # Global latent representation from input image
        >>> image_latent = LatentVariable(
        ...     concepts='global_image_features',
        ...     distribution=Delta,
        ...     size=512  # 512-dimensional global latent
        ... )
        >>> 
        >>> # Multiple latent variables for hierarchical representation
        >>> low_level_features = LatentVariable(
        ...     concepts='low_level_features',
        ...     distribution=Delta,
        ...     size=256
        ... )
        >>> high_level_features = LatentVariable(
        ...     concepts='high_level_features',
        ...     distribution=Delta,
        ...     size=512
        ... )
    """
    
    def __init__(self, concept: Optional[str] = None,
                 concepts: Optional[List[str]] = None,
                 distribution: Union[Type[Distribution], List[Type[Distribution]]] = None,
                 size: Union[int, List[int]] = 1,
                 metadata: Dict[str, Any] = None,
                 dist_kwargs: Optional[Dict[str, Any]] = None,
                 observed: bool = False,
                 **kwargs):
        if metadata is None:
            metadata = {}
        metadata['variable_type'] = 'latent'
        super().__init__(concept=concept, concepts=concepts,
                         distribution=distribution, size=size,
                         metadata=metadata, dist_kwargs=dist_kwargs,
                         observed=observed, **kwargs)


# Backward compatibility alias
InputVariable = LatentVariable
