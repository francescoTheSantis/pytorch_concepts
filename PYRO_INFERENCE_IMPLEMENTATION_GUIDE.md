# Guide: Implementing Pyro-Based Inference in torch_concepts

## Overview

This document provides comprehensive guidelines for implementing probabilistic inference strategies using Pyro in the torch_concepts library. It covers the architecture, design patterns, and step-by-step implementation process.

---

## Table of Contents

1. [User Request](#user-request)
2. [Library Architecture](#library-architecture)
3. [Inference System Overview](#inference-system-overview)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Pyro Integration Patterns](#pyro-integration-patterns)
6. [Example: ExactInference Implementation](#example-exactinference-implementation)
7. [Testing and Validation](#testing-and-validation)
8. [Common Pitfalls](#common-pitfalls)

---

## User Request

**Original Request:**
> "Implement an exact inference method that combines the capabilities of CBMs (Concept Bottleneck Models) with Probabilistic modeling using Pyro, a well-known library for probabilistic programming."

**Key Requirements:**
- Perform exact marginal inference on trained CBMs
- Support arbitrary marginal queries: `p(c_i | evidence, x)`
- Support conditional queries: `p(c_i | c_j, x)`
- Integrate with existing inference architecture
- Use Pyro for probabilistic computations

---

## Library Architecture

### Three Abstraction Levels

The torch_concepts library is organized into three levels:

```
torch_concepts/nn/modules/
├── low/          # Fundamental building blocks
│   └── base/
│       ├── inference.py      # BaseInference, BaseIntervention
│       └── ...
├── mid/          # Composition and inference strategies
│   ├── inference/
│   │   ├── forward.py        # ForwardInference (base for most strategies)
│   │   ├── deterministic.py  # DeterministicInference
│   │   ├── ancestral.py      # AncestralSamplingInference
│   │   ├── independent.py    # IndependentInference
│   │   └── exact.py          # ExactInference (NEW - Pyro-based)
│   └── ...
└── high/         # End-to-end models (CBM, CEM, etc.)
```

### Key Components

1. **ProbabilisticModel**: Represents a directed acyclic graph (DAG) of variables with conditional probability distributions
2. **BaseInference**: Abstract base class for all inference strategies
3. **ForwardInference**: Implements topological sorting and parallel execution
4. **ParametricCPD**: Neural network parameterizations of conditional probability distributions

---

## Inference System Overview

### BaseInference Abstract Class

All inference strategies must inherit from `BaseInference`:

```python
class BaseInference(nn.Module, ABC):
    """Abstract base class for inference strategies."""
    
    def __init__(self, model: ProbabilisticModel):
        super().__init__()
        self.model = model
    
    @abstractmethod
    def get_results(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        intervention: Optional[Intervention] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform inference and return results for queried variables.
        
        Args:
            query: List of variable names to query
            evidence: Dictionary mapping variable names to observed values
            intervention: Optional intervention object
            **kwargs: Additional inference-specific arguments
        
        Returns:
            Dictionary mapping variable names to output tensors
        """
        pass
    
    def query(self, query: List[str], evidence: Dict[str, torch.Tensor], **kwargs):
        """Convenience method that returns concatenated results."""
        results = self.get_results(query, evidence, **kwargs)
        return torch.cat([results[var] for var in query], dim=-1)
```

### Design Patterns

1. **Topological Sorting**: Variables are processed in dependency order
2. **Evidence Dictionary**: Input format is `{variable_name: tensor}`
3. **Query List**: Output variables specified as list of names
4. **Lazy Evaluation**: Only compute what's needed for the query
5. **Tensor Reshaping**: Handle both single samples and batches

---

## Implementation Guidelines

### Step 1: Understand the Model Structure

A `ProbabilisticModel` contains:

```python
model.variables           # List[Variable] - all variables in the DAG
model.variable_dict       # Dict[str, Variable] - lookup by name
model.parametric_cpds     # ModuleList[ParametricCPD] - neural networks
model.graph               # NetworkX DiGraph - DAG structure
model._topo_order         # List[str] - topologically sorted variable names
```

Each `Variable` has:
- `name`: Variable identifier
- `parents`: List of parent variable names
- `distribution`: PyTorch distribution class (Bernoulli, Categorical, etc.)
- `size`: Dimensionality

### Step 2: Choose Base Class

**Option A: Inherit from ForwardInference**
- ✅ Get topological sorting for free
- ✅ Automatic parallel execution of independent paths
- ✅ Built-in evidence handling
- ⚠️ Must implement `_compute_variable()`

**Option B: Inherit directly from BaseInference**
- ✅ Full control over inference process
- ⚠️ Must handle topological ordering yourself
- ⚠️ Must implement `get_results()`

**Recommendation**: Inherit from `ForwardInference` for most cases.

### Step 3: Implement Required Methods

#### If inheriting from ForwardInference:

```python
class MyInference(ForwardInference):
    def _compute_variable(
        self,
        var: Variable,
        parent_values: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute output for a single variable.
        
        Args:
            var: Variable to compute
            parent_values: Dict of already-computed parent values
            evidence: Original evidence dictionary
            **kwargs: Additional arguments
        
        Returns:
            Tensor of shape (batch_size, var.size)
        """
        # Your inference logic here
        pass
```

### Step 4: Handle Distribution Types

The library supports various PyTorch distributions:

```python
from torch.distributions import (
    Bernoulli,           # Binary concepts
    Categorical,         # Multi-class concepts
    RelaxedBernoulli,    # Continuous relaxation of Bernoulli
    RelaxedOneHotCategorical,  # Continuous relaxation of Categorical
)
```

**Key Pattern**: Always work with distribution parameters (logits), not samples:

```python
# Get parameters from neural network
cpd = self.model.get_cpd(var.name)
params = cpd(parent_values)  # Returns dict with 'logits' or 'probs'

# Create distribution
dist = var.distribution(**params)

# Get mean or sample
if deterministic:
    return dist.mean  # or dist.probs for categorical
else:
    return dist.sample()
```

---

## Pyro Integration Patterns

### Pattern 1: Pyro Model Construction

Convert ProbabilisticModel to Pyro model:

```python
def _build_pyro_model(self, evidence: Dict[str, torch.Tensor]):
    """Build Pyro model from ProbabilisticModel."""
    
    # Create Pyro model function
    def model():
        # Dictionary to store sampled values
        values = {}
        
        # Start with evidence
        values.update(evidence)
        
        # Process variables in topological order
        for var_name in self.model._topo_order:
            var = self.model.variable_dict[var_name]
            
            # Skip if already in evidence
            if var_name in evidence:
                continue
            
            # Get parent values
            parent_vals = {p: values[p] for p in var.parents if p in values}
            
            if not parent_vals:
                # No parents or parents not available
                continue
            
            # Get CPD and compute parameters
            cpd = self.model.get_cpd(var_name)
            params = cpd(parent_vals)
            
            # Create Pyro distribution and sample
            if var.distribution == Bernoulli:
                values[var_name] = pyro.sample(
                    var_name,
                    dist.Bernoulli(logits=params['logits'])
                )
            elif var.distribution == Categorical:
                values[var_name] = pyro.sample(
                    var_name,
                    dist.Categorical(logits=params['logits'])
                )
            # ... handle other distributions
        
        return values
    
    return model
```

### Pattern 2: Importance Sampling for Marginals

```python
def marginal(
    self,
    query: List[str],
    evidence: Dict[str, torch.Tensor],
    num_samples: int = 1000
) -> torch.Tensor:
    """
    Compute marginal probabilities using importance sampling.
    
    Args:
        query: Variables to marginalize
        evidence: Observed variables
        num_samples: Number of importance samples
    
    Returns:
        Tensor of marginal probabilities
    """
    batch_size = next(iter(evidence.values())).shape[0]
    
    # Build Pyro model
    pyro_model = self._build_pyro_model(evidence)
    
    # Perform importance sampling
    marginals = []
    for i in range(batch_size):
        # Extract single sample evidence
        sample_evidence = {k: v[i:i+1] for k, v in evidence.items()}
        
        # Run importance sampling
        importance = pyro.infer.Importance(pyro_model, num_samples=num_samples)
        importance.run()
        
        # Estimate marginals
        marginal_probs = self._estimate_marginals(importance, query)
        marginals.append(marginal_probs)
    
    return torch.cat(marginals, dim=0)
```

### Pattern 3: Handling Continuous vs Discrete Distributions

```python
def _get_pyro_distribution(self, var: Variable, params: Dict):
    """Convert torch distribution to Pyro distribution."""
    
    if var.distribution == Bernoulli:
        return dist.Bernoulli(logits=params['logits'])
    
    elif var.distribution == RelaxedBernoulli:
        # Use Bernoulli for discrete sampling in Pyro
        return dist.Bernoulli(logits=params['logits'])
    
    elif var.distribution == Categorical:
        return dist.Categorical(logits=params['logits'])
    
    elif var.distribution == RelaxedOneHotCategorical:
        # Use Categorical for discrete sampling in Pyro
        return dist.Categorical(logits=params['logits'])
    
    else:
        raise ValueError(f"Unsupported distribution: {var.distribution}")
```

---

## Example: ExactInference Implementation

### File Structure

```
torch_concepts/nn/modules/mid/inference/exact.py
```

### Complete Implementation

```python
"""
Exact Marginal Inference using Pyro

This module implements exact inference for probabilistic concept-based models
using Pyro's importance sampling.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

# Pyro imports (required dependency)
import pyro
import pyro.distributions as dist
from pyro.infer import Importance, EmpiricalMarginal

from .forward import ForwardInference
from ..low.base import ProbabilisticModel, Variable


class ExactInference(ForwardInference):
    """
    Exact marginal inference using Pyro importance sampling.
    
    This inference strategy allows computing arbitrary marginal probabilities
    p(c_i | evidence, x) for any subset of concept variables, including
    conditional queries p(c_i | c_j, x).
    
    Key Features:
    - Exact marginal computation via importance sampling
    - Support for arbitrary evidence (both inputs and concepts)
    - Conditional probability queries
    - Integration with trained CBMs
    
    Args:
        model: ProbabilisticModel to perform inference on
        num_samples: Number of importance samples for marginal estimation
    
    Example:
        >>> # Train a CBM
        >>> inference = DeterministicInference(model)
        >>> # ... training code ...
        
        >>> # Switch to exact inference
        >>> exact_inf = ExactInference(model, num_samples=1000)
        >>> 
        >>> # Query marginal: p(concept1 | x)
        >>> p_c1 = exact_inf.marginal(['concept1'], evidence={'input': x})
        >>> 
        >>> # Conditional query: p(concept2 | concept1=1, x)
        >>> p_c2_given_c1 = exact_inf.marginal(
        ...     ['concept2'],
        ...     evidence={'input': x, 'concept1': torch.ones(batch_size, 1)}
        ... )
    """
    
    def __init__(
        self,
        model: ProbabilisticModel,
        num_samples: int = 1000,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.num_samples = num_samples
    
    def marginal(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute marginal probabilities using Pyro importance sampling.
        
        This method computes p(query | evidence) by:
        1. Building a Pyro probabilistic model
        2. Running importance sampling
        3. Estimating marginals from weighted samples
        
        Args:
            query: List of variable names to compute marginals for
            evidence: Dictionary of observed variables (inputs + concepts)
            num_samples: Number of importance samples (overrides default)
        
        Returns:
            Tensor of marginal probabilities, shape (batch_size, sum of query sizes)
            For Bernoulli variables: returns p(variable=1)
            For Categorical variables: returns probabilities for each class
        
        Example:
            >>> # Single concept marginal
            >>> p_engine = exact_inf.marginal(['engine'], {'input': x})
            >>> # Shape: (batch_size, 1)
            
            >>> # Multiple concepts
            >>> p_concepts = exact_inf.marginal(['engine', 'wheels'], {'input': x})
            >>> # Shape: (batch_size, 2)
            
            >>> # Conditional
            >>> p_task = exact_inf.marginal(
            ...     ['car_start'],
            ...     {'input': x, 'engine': torch.ones(batch, 1)}
            ... )
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        # Get batch size from evidence
        batch_size = next(iter(evidence.values())).shape[0]
        
        # Process each sample in batch separately (importance sampling is per-sample)
        all_marginals = []
        
        for i in range(batch_size):
            # Extract single sample evidence
            sample_evidence = {k: v[i:i+1] for k, v in evidence.items()}
            
            # Build Pyro model for this sample
            pyro_model = self._build_pyro_model(sample_evidence)
            
            # Run importance sampling
            importance = Importance(pyro_model, num_samples=num_samples)
            importance.run()
            
            # Estimate marginals for queried variables
            marginal_results = []
            for var_name in query:
                var = self.model.variable_dict[var_name]
                
                # Get empirical marginal from importance samples
                marginal_dist = EmpiricalMarginal(importance, sites=var_name)
                
                # Estimate probability
                if var.distribution in [torch.distributions.Bernoulli,
                                       torch.distributions.RelaxedBernoulli]:
                    # For binary: estimate p(var=1)
                    samples = torch.stack([marginal_dist.sample() for _ in range(100)])
                    prob = samples.float().mean(dim=0, keepdim=True)
                    marginal_results.append(prob)
                
                elif var.distribution in [torch.distributions.Categorical,
                                         torch.distributions.RelaxedOneHotCategorical]:
                    # For categorical: estimate probability for each class
                    samples = torch.stack([marginal_dist.sample() for _ in range(100)])
                    # samples shape: (100, var.size)
                    probs = samples.float().mean(dim=0, keepdim=True)
                    marginal_results.append(probs)
                
                else:
                    raise ValueError(f"Unsupported distribution for {var_name}")
            
            # Concatenate marginals for this sample
            sample_marginal = torch.cat(marginal_results, dim=-1)
            all_marginals.append(sample_marginal)
        
        # Stack all batch samples
        return torch.cat(all_marginals, dim=0)
    
    def _build_pyro_model(self, evidence: Dict[str, torch.Tensor]):
        """
        Build Pyro probabilistic model from ProbabilisticModel.
        
        Args:
            evidence: Observed variables (fixed during sampling)
        
        Returns:
            Callable Pyro model function
        """
        model = self.model
        
        def pyro_model():
            # Dictionary to accumulate sampled values
            values = {}
            
            # Add evidence (these are observed)
            values.update(evidence)
            
            # Process variables in topological order
            for var_name in model._topo_order:
                var = model.variable_dict[var_name]
                
                # Skip if in evidence (already observed)
                if var_name in evidence:
                    continue
                
                # Check if all parents are available
                parent_vals = {}
                for parent_name in var.parents:
                    if parent_name in values:
                        parent_vals[parent_name] = values[parent_name]
                    else:
                        # Parent not available, skip this variable
                        break
                
                if len(parent_vals) != len(var.parents):
                    # Not all parents available
                    continue
                
                # Get CPD and compute distribution parameters
                cpd = model.get_cpd(var_name)
                params = cpd(parent_vals)
                
                # Create Pyro distribution and sample
                if var.distribution in [torch.distributions.Bernoulli,
                                       torch.distributions.RelaxedBernoulli]:
                    # Use Bernoulli for discrete sampling
                    pyro_dist = dist.Bernoulli(logits=params['logits'])
                    values[var_name] = pyro.sample(var_name, pyro_dist)
                
                elif var.distribution in [torch.distributions.Categorical,
                                         torch.distributions.RelaxedOneHotCategorical]:
                    # Use Categorical for discrete sampling
                    pyro_dist = dist.Categorical(logits=params['logits'])
                    # Sample returns class index, convert to one-hot
                    sample_idx = pyro.sample(var_name, pyro_dist)
                    one_hot = torch.nn.functional.one_hot(
                        sample_idx.long(),
                        num_classes=var.size
                    ).float()
                    values[var_name] = one_hot
                
                else:
                    raise ValueError(f"Unsupported distribution: {var.distribution}")
            
            return values
        
        return pyro_model
```

### Integration with Module System

Update `__init__.py`:

```python
# torch_concepts/nn/modules/mid/inference/__init__.py

from .forward import ForwardInference
from .deterministic import DeterministicInference
from .ancestral import AncestralSamplingInference
from .independent import IndependentInference
from .exact import ExactInference

__all__ = [
    'ForwardInference',
    'DeterministicInference',
    'AncestralSamplingInference',
    'IndependentInference',
    'ExactInference',
]
```

---

## Testing and Validation

### Unit Tests

```python
# tests/nn/inference/test_exact_inference.py

import torch
from torch_concepts import ExactInference, ProbabilisticModel
from torch_concepts.data.datasets import ToyDAGDataset

def test_exact_inference_basic():
    """Test basic ExactInference functionality."""
    # Create dataset
    dataset = ToyDAGDataset(
        variables=['a', 'b', 'c'],
        cardinalities={'a': 2, 'b': 2, 'c': 2},
        dag=[('a', 'c'), ('b', 'c')],
        conditional_probs={...},
        n_gen=100
    )
    
    # Create and train model
    model = create_model()
    # ... training code ...
    
    # Test exact inference
    exact_inf = ExactInference(model, num_samples=1000)
    
    # Test marginal query
    evidence = {'input': dataset.input_data[:10]}
    marginal = exact_inf.marginal(['a'], evidence)
    
    assert marginal.shape == (10, 1)
    assert torch.all((marginal >= 0) & (marginal <= 1))

def test_conditional_query():
    """Test conditional probability queries."""
    exact_inf = ExactInference(model)
    
    # p(b | a=1, x)
    evidence = {
        'input': x,
        'a': torch.ones(batch_size, 1)
    }
    marginal = exact_inf.marginal(['b'], evidence)
    
    assert marginal.shape == (batch_size, 1)
```

### Example Script

Create comprehensive example (as shown in the actual implementation).

---

## Common Pitfalls

### 1. **Tensor Shape Mismatches**

❌ **Wrong:**
```python
# Forgetting batch dimension
evidence = {'input': torch.randn(10)}  # Missing batch dim
```

✅ **Correct:**
```python
evidence = {'input': torch.randn(batch_size, 10)}  # (batch, features)
```

### 2. **Distribution Parameter Formats**

❌ **Wrong:**
```python
# Using probabilities instead of logits
dist = Bernoulli(probs=params)  # Inconsistent with library
```

✅ **Correct:**
```python
# Always use logits
dist = Bernoulli(logits=params['logits'])
```

### 3. **Evidence Dictionary Keys**

❌ **Wrong:**
```python
# Wrong variable names
evidence = {'x': input_tensor}  # Should match model variable names
```

✅ **Correct:**
```python
# Use exact variable names from model
evidence = {'input': input_tensor}  # Matches LatentVariable("input", ...)
```

### 4. **One-Hot Encoding**

❌ **Wrong:**
```python
# Binary concept as single value
concept = torch.tensor([0, 1, 1, 0])  # Shape: (4,)
```

✅ **Correct:**
```python
# Binary concept with feature dimension
concept = torch.tensor([[0], [1], [1], [0]])  # Shape: (4, 1)
```

### 5. **Pyro State Management**

❌ **Wrong:**
```python
# Not clearing Pyro state between calls
for i in range(batch_size):
    importance.run()  # State accumulates!
```

✅ **Correct:**
```python
# Clear state or create new inference object
for i in range(batch_size):
    pyro.clear_param_store()
    importance = Importance(model, num_samples=n)
    importance.run()
```

---

## Best Practices

### 1. **Start Simple**

Begin with a minimal working example:
- Single Bernoulli concept
- Small batch size (1-10 samples)
- Few importance samples (100-500)
- Print intermediate shapes

### 2. **Validate Against Deterministic**

Compare Pyro inference with deterministic inference:
```python
# Should agree when model is well-trained
det_inf = DeterministicInference(model)
exact_inf = ExactInference(model, num_samples=10000)

det_probs = torch.sigmoid(det_inf.query(['concept'], evidence))
exact_probs = exact_inf.marginal(['concept'], evidence)

assert torch.allclose(det_probs, exact_probs, atol=0.1)
```

### 3. **Profile Performance**

Exact inference can be slow:
- Use fewer samples during development
- Process batches in parallel when possible
- Consider caching for repeated queries

### 4. **Document Query Formats**

Make it clear what the output represents:
```python
# Good documentation
"""
Returns:
    For Bernoulli: p(variable=1), shape (batch, 1)
    For Categorical: [p(class_0), ..., p(class_K)], shape (batch, K)
"""
```

### 5. **Handle Edge Cases**

```python
# Check for empty evidence
if not evidence:
    raise ValueError("Evidence dictionary cannot be empty")

# Check for invalid queries
for var_name in query:
    if var_name not in self.model.variable_dict:
        raise ValueError(f"Variable '{var_name}' not in model")
```

---

## Summary Checklist

When implementing a new Pyro-based inference strategy:

- [ ] Choose appropriate base class (ForwardInference recommended)
- [ ] Implement required abstract methods
- [ ] Handle all distribution types used in the model
- [ ] Support batched inputs
- [ ] Convert between torch and Pyro distributions correctly
- [ ] Manage Pyro state appropriately
- [ ] Add to `__init__.py` exports
- [ ] Write unit tests
- [ ] Create example script demonstrating usage
- [ ] Document query input/output formats
- [ ] Validate against existing inference methods
- [ ] Profile performance on realistic datasets

---

## Additional Resources

- **Pyro Documentation**: https://pyro.ai/
- **Importance Sampling Guide**: https://pyro.ai/examples/inference.html
- **torch_concepts Examples**: `examples/utilization/1_pgm/`
- **Existing Inference Implementations**: `torch_concepts/nn/modules/mid/inference/`

---

## Contact & Support

For questions or issues:
1. Check existing inference implementations as reference
2. Review ToyDAGDataset examples for model creation
3. Test with simple DAG structures first
4. Validate tensor shapes at each step

Good luck with your implementation! 🚀
