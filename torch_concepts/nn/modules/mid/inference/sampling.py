"""Sampling-based marginal inference using Pyro for probabilistic models.

This module provides approximate marginal inference for trained concept-based models
by leveraging Pyro's importance sampling. It enables arbitrary marginal queries 
conditioned on input evidence using Monte Carlo approximation.

Algorithm: Importance Sampling (Monte Carlo approximation)
- NOT exact inference - estimates marginals by averaging over samples
- Accuracy improves with more samples: error ~ O(1/√num_samples)

Requires: pyro-ppl (install with: pip install pyro-ppl)
"""

import torch
from typing import List, Dict, Optional, Union

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import Importance, EmpiricalMarginal
from pyro.infer.enum import config_enumerate

from ...low.base.inference import BaseInference
from ..models.variable import Variable, ConceptVariable, LatentVariable
from ..models.probabilistic_model import ProbabilisticModel
from ...low.base.graph import BaseGraphLearner
from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical


class SamplingInference(BaseInference):
        """
        Sampling-based marginal inference using Pyro importance sampling.
        
        This inference engine performs approximate marginal inference over discrete concepts
        by converting the trained ProbabilisticModel into a Pyro model and using
        importance sampling. It supports arbitrary conditional queries of the
        form p(c_i | evidence, x) where evidence contains observed concepts.
        
        **Algorithm:** Importance Sampling (Monte Carlo approximation)
        - Draws num_samples from the generative model
        - Estimates marginals by averaging samples
        
        **Key Features:**
        - Approximate marginal computation for discrete concepts
        - Supports arbitrary conditioning on concepts and inputs
        - **Batched operations** for efficient inference across multiple samples
        - No gradient computation (inference only)
        - Compatible with all trained CBM models
        - Handles Bernoulli and Categorical distributions
        
        **Performance:**
        - Uses Pyro's plate notation for vectorized batch processing
        - Processes entire batches simultaneously (no per-sample loops)
        - Significantly faster than per-sample inference
        
        **Limitations:**
        - Only works with discrete distributions (Bernoulli, Categorical)
        - All queries must be conditioned on input x
        - Computational cost grows linearly with num_samples
        - Approximation quality depends on num_samples
        - Does not support continuous distributions
        
        Args:
            probabilistic_model: The trained ProbabilisticModel to perform inference on.
            graph_learner: Optional graph learner for weighted adjacency structure.
            num_samples: Number of importance samples for marginal estimation (default: 1000).
                More samples = better approximation but slower inference.
            
        Example:
            >>> import torch
            >>> from torch.distributions import Bernoulli
            >>> from torch_concepts import LatentVariable, ConceptVariable
            >>> from torch_concepts.distributions import Delta
            >>> from torch_concepts.nn import SamplingInference, ParametricCPD, ProbabilisticModel
            >>> from torch_concepts.nn import LinearLatentToConcept, LinearConceptToConcept
            >>>
            >>> # Assume we have a trained model: input -> c1 -> c2 -> task
            >>> # (See example script for full training code)
            >>>
            >>> # Create sampling inference engine
            >>> inference = SamplingInference(trained_pgm, num_samples=2000)
            >>>
            >>> # Query marginal p(c1 | x) - processes entire batch at once!
            >>> x = torch.randn(32, 10)  # Batch of 32 samples
            >>> p_c1 = inference.marginal(['c1'], evidence={'input': x})
            >>> # Returns tensor of shape (32, 1) with approximate probabilities
            >>>
            >>> # Query conditional p(task | c1=1, x)
            >>> c1_observed = torch.ones(32, 1)
            >>> p_task_given_c1 = inference.marginal(
            ...     ['task'], 
            ...     evidence={'input': x, 'c1': c1_observed}
            ... )
            >>>
            >>> # Query joint p(c1, c2 | x)
            >>> p_c1_c2 = inference.marginal(['c1', 'c2'], evidence={'input': x})
            >>> # Returns tensor of shape (32, 2) with joint probabilities
            >>>
            >>> # Forward-style API (query all concepts)
            >>> all_probs = inference(x)  # or inference.query(x)
            >>>
            >>> # Compare with deterministic inference (for validation)
            >>> from torch_concepts.nn import DeterministicInference
            >>> det_inference = DeterministicInference(trained_pgm)
            >>> logits = det_inference.query(['c1'], evidence={'input': x})
            >>> probs = torch.sigmoid(logits)
            >>> # p_c1 and probs should be similar for well-trained models
        """
        
        def __init__(
            self,
            probabilistic_model: ProbabilisticModel,
            graph_learner: BaseGraphLearner = None,
            num_samples: int = 1000,
        ):
            super().__init__()
            self.probabilistic_model = probabilistic_model
            self.graph_learner = graph_learner
            self.num_samples = num_samples
            
            # Build variable map and topological order
            self.variable_map = {var.concept: var for var in probabilistic_model.variables}
            self._topo_order = self._compute_topological_order()
            
            # Validate that all ConceptVariables use discrete distributions
            self._validate_discrete_distributions()
            
        def _compute_topological_order(self) -> List[str]:
            """Compute topological ordering of variables using DFS."""
            visited = set()
            order = []
            
            def dfs(var_name: str):
                if var_name in visited:
                    return
                visited.add(var_name)
                
                var = self.variable_map[var_name]
                for parent in var.parents:
                    if isinstance(parent, Variable):
                        dfs(parent.concept)
                    else:
                        dfs(parent)
                
                order.append(var_name)
            
            for var_name in self.variable_map:
                dfs(var_name)
            
            return order
            
        def _validate_discrete_distributions(self):
            """Validate that all concept variables use discrete distributions."""
            
            supported_distributions = {Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical}
            
            for var in self.probabilistic_model.variables:
                if isinstance(var, ConceptVariable):
                    if var.distribution not in supported_distributions:
                        raise ValueError(
                            f"SamplingInference only supports discrete distributions. "
                            f"Variable '{var.concept}' uses {var.distribution.__name__} "
                            f"which is not supported. Supported: Bernoulli, Categorical."
                        )
        
        @property
        def query_kwargs(self) -> frozenset:
            """Return the set of keyword argument names accepted by query."""
            return frozenset(['return_logits', 'num_samples'])
        
        def ground_truth_to_evidence(self, value: torch.Tensor, cardinality: int) -> torch.Tensor:
            """
            Convert ground truth to probability format for sampling inference.
            
            Parameters
            ----------
            value : torch.Tensor
                Ground truth indices. Shape: (batch_size,) or (batch_size, 1).
            cardinality : int
                Number of classes (1 for binary, >1 for categorical).
                
            Returns
            -------
            torch.Tensor
                Probability format (one-hot for observed, or probabilities).
            """
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            
            if cardinality > 1:
                # Categorical: one-hot encoding
                return torch.nn.functional.one_hot(
                    value.squeeze(-1).long(),
                    num_classes=cardinality
                ).float()
            else:
                # Binary: 0 or 1 as probability
                return value.float()
        
        def _build_pyro_model(self, evidence: Dict[str, torch.Tensor], batch_size: int):
            """
            Build a batched Pyro model that processes all samples simultaneously.
            
            This creates a vectorized generative model using pyro.plate to handle
            the batch dimension efficiently, avoiding per-sample loops.
            
            Args:
                evidence: Dictionary of observed variables (including 'input').
                batch_size: Number of samples in the batch.
                
            Returns:
                Callable Pyro model that returns dictionary of sampled values.
            """
            def pyro_model():
                # Use pyro.plate for batch dimension vectorization
                with pyro.plate("batch", batch_size, dim=-2):
                    with torch.no_grad():
                        results = {}
                        
                        # Process variables in topological order
                        for var_name in self._topo_order:
                            var = self.variable_map[var_name]
                            
                            # If observed in evidence, use observed value
                            if var_name in evidence:
                                results[var_name] = evidence[var_name]
                                continue
                            
                            # Get CPD
                            parametric_cpd = self.probabilistic_model.get_module_of_concept(var_name)
                            if parametric_cpd is None:
                                raise RuntimeError(f"Missing CPD for concept: {var_name}")
                            
                            # Compute parent inputs
                            parent_inputs = self._get_parent_inputs(var, results, evidence)
                            
                            # Compute CPD output (logits) - batched
                            logits = parametric_cpd.forward(**parent_inputs)
                            
                            # Sample from Pyro distribution (vectorized over batch)                            
                            if isinstance(var, LatentVariable):
                                # Latent variables are deterministic
                                results[var_name] = logits
                            elif var.distribution in [Bernoulli, RelaxedBernoulli] or var.size == 1:
                                # Binary concept - batched sampling
                                probs = torch.sigmoid(logits)  # Shape: (batch_size, 1)
                                sample = pyro.sample(
                                    var_name,
                                    dist.Bernoulli(probs.squeeze(-1)).to_event(0)
                                )
                                results[var_name] = sample.unsqueeze(-1)  # Add feature dim back
                            elif var.distribution in [Categorical, RelaxedOneHotCategorical] or var.size > 1:
                                # Categorical concept - batched sampling
                                probs = torch.softmax(logits, dim=-1)  # Shape: (batch_size, num_classes)
                                sample = pyro.sample(
                                    var_name,
                                    dist.Categorical(probs).to_event(0)
                                )
                                # Convert to one-hot
                                sample_onehot = torch.nn.functional.one_hot(
                                    sample.long(), num_classes=var.size
                                ).float()
                                results[var_name] = sample_onehot
                            else:
                                raise ValueError(f"Unsupported distribution for {var_name}")
                        
                        return results
            
            return pyro_model
        
        def _get_parent_inputs(
            self, 
            var: Variable, 
            results: Dict[str, torch.Tensor],
            evidence: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """
            Gather parent inputs for a variable's CPD.
            
            Args:
                var: The variable to compute inputs for.
                results: Dictionary of already-computed values.
                evidence: Original evidence dictionary.
                
            Returns:
                Dictionary of keyword arguments for the CPD forward method.
            """
            if not var.parents:
                # Root node - should be in evidence
                if var.concept not in evidence:
                    raise ValueError(f"Root variable '{var.concept}' must be in evidence")
                return {'latent': evidence[var.concept]}
            
            # Collect parent values
            parent_concepts = []
            parent_latents = []
            
            for parent in var.parents:
                parent_name = parent.concept if isinstance(parent, Variable) else parent
                
                if parent_name not in results:
                    raise RuntimeError(f"Parent {parent_name} not computed yet for {var.concept}")
                
                parent_var = self.variable_map[parent_name]
                parent_value = results[parent_name]
                
                # Apply graph learner weights if available
                if isinstance(parent_var, ConceptVariable) and self.graph_learner is not None:
                    # Get edge weight from graph learner
                    parent_idx = list(self.variable_map.keys()).index(parent_name)
                    child_idx = list(self.variable_map.keys()).index(var.concept)
                    weight = self.graph_learner.weighted_adj[parent_idx, child_idx]
                    parent_value = parent_value * weight
                
                # Categorize parent type
                if isinstance(parent_var, LatentVariable):
                    parent_latents.append(parent_value)
                else:
                    parent_concepts.append(parent_value)
            
            # Build kwargs dictionary
            kwargs = {}
            if parent_latents:
                kwargs['latent'] = torch.cat(parent_latents, dim=-1) if len(parent_latents) > 1 else parent_latents[0]
            if parent_concepts:
                kwargs['concept'] = torch.cat(parent_concepts, dim=-1) if len(parent_concepts) > 1 else parent_concepts[0]
            
            return kwargs
        
        def marginal(
            self,
            query: List[str],
            evidence: Dict[str, torch.Tensor],
            num_samples: Optional[int] = None,
            **kwargs
        ) -> torch.Tensor:
            """
            Compute approximate marginal probabilities using importance sampling.
            
            This method performs batched importance sampling by:
            1. Building a vectorized Pyro model for the entire batch
            2. Drawing num_samples from the model with batched operations
            3. Estimating marginals by averaging samples (Monte Carlo approximation)
            
            Args:
                query: List of concept names to compute marginals for.
                evidence: Dictionary of observed variables (must include 'input').
                num_samples: Number of importance samples (overrides default).
                    More samples = better approximation but slower.
                **kwargs: Additional arguments (unused, for compatibility).
                
            Returns:
                torch.Tensor: Approximate marginal probabilities for queried concepts.
                    - Binary (size=1): Shape (batch, 1) with values in [0, 1]
                    - Categorical (size>1): Shape (batch, size) with probabilities
                    
            Note:
                Results are approximate! Accuracy improves with num_samples:
                - 100 samples: ~10% error
                - 1000 samples: ~3% error
                - 10000 samples: ~1% error
                    
            Example:
                >>> # Query p(c1 | x) - now processes entire batch at once!
                >>> p_c1 = inference.marginal(['c1'], evidence={'input': x})
                >>>
                >>> # Query p(c2 | c1=1, x)
                >>> p_c2_given_c1 = inference.marginal(
                ...     ['c2'], 
                ...     evidence={'input': x, 'c1': torch.ones(batch_size, 1)}
                ... )
            """
            self._validate_evidence(evidence)
            
            if not query:
                raise ValueError("Query list cannot be empty")
            
            if num_samples is None:
                num_samples = self.num_samples
            
            batch_size = evidence['input'].shape[0]
            
            # Build batched Pyro model
            pyro_model_fn = self._build_pyro_model(evidence, batch_size)
            
            # Collect marginals for all queried concepts
            all_marginals = []
            
            for concept_name in query:
                if concept_name not in self.variable_map:
                    raise ValueError(f"Query concept '{concept_name}' not found in model")
                
                var = self.variable_map[concept_name]
                
                # If concept is in evidence, use observed value
                if concept_name in evidence:
                    obs_value = evidence[concept_name]
                    all_marginals.append(obs_value)
                    continue
                
                # Run importance sampling (batched)
                importance = Importance(pyro_model_fn, num_samples=num_samples)
                importance.run()
                
                # Get empirical marginal
                marginal_dist = EmpiricalMarginal(importance, sites=[concept_name])
                
                # Sample from empirical distribution to estimate marginal
                # This gives us (num_samples, batch_size, *var_shape)
                samples = torch.stack([marginal_dist.sample() for _ in range(num_samples)])
                
                # Compute expected value over samples (Monte Carlo estimate)
                # Result shape: (batch_size, *var_shape)
                if var.size == 1:
                    # Binary: compute mean probability
                    # samples shape: (num_samples, batch_size, 1) or (num_samples, batch_size)
                    if samples.dim() == 2:
                        samples = samples.unsqueeze(-1)
                    mean_prob = samples.float().mean(dim=0)  # (batch_size, 1)
                    all_marginals.append(mean_prob)
                else:
                    # Categorical: compute probability distribution
                    # samples shape: (num_samples, batch_size, num_classes)
                    mean_probs = samples.float().mean(dim=0)  # (batch_size, num_classes)
                    all_marginals.append(mean_probs)
            
            # Concatenate marginals for all queried concepts
            return torch.cat(all_marginals, dim=-1)
        
        def query(
            self,
            query_vars: Union[List[str], torch.Tensor],
            evidence: Optional[Dict[str, torch.Tensor]] = None,
            return_logits: bool = False,
            num_samples: Optional[int] = None,
            **kwargs
        ) -> torch.Tensor:
            """
            Query approximate marginal probabilities (main API method).
            
            This method provides the BaseInference-compatible API and supports
            both evidence dictionary and direct input tensor formats.
            
            Args:
                query_vars: Either a list of concept names to query, or input tensor x.
                    If tensor, queries all concept variables given x as input.
                evidence: Dictionary of observed variables. If None and query_vars is a
                    tensor, creates evidence={'input': query_vars}.
                return_logits: If True, convert probabilities back to logits.
                num_samples: Number of importance samples (overrides default).
                **kwargs: Additional arguments (unused, for compatibility).
                
            Returns:
                torch.Tensor: Approximate marginal probabilities (or logits if return_logits=True).
                
            Example:
                >>> # Method 1: Query with evidence dictionary
                >>> probs = inference.query(['c1', 'c2'], evidence={'input': x})
                >>>
                >>> # Method 2: Query all concepts with direct input (forward-style)
                >>> probs = inference.query(x)  # Same as inference(x)
                >>>
                >>> # Method 3: Conditional query
                >>> probs = inference.query(
                ...     ['c2'], 
                ...     evidence={'input': x, 'c1': torch.ones(batch, 1)}
                ... )
            """
            # Handle case where query_vars is actually the input tensor
            if isinstance(query_vars, torch.Tensor):
                # Forward-style call: query(x)
                x = query_vars
                evidence = {'input': x}
                # Query all concept variables
                query_list = [v.concept for v in self.probabilistic_model.variables 
                             if isinstance(v, ConceptVariable)]
            else:
                # Standard call: query(['c1', 'c2'], evidence={...})
                query_list = query_vars
                if evidence is None:
                    raise ValueError("Evidence dictionary required when querying specific concepts")
            
            # Compute marginals
            marginals = self.marginal(query_list, evidence, num_samples=num_samples, **kwargs)
            
            if return_logits:
                # Convert probabilities back to logits for compatibility
                # Clamp to avoid log(0) or log(1)
                marginals_clamped = torch.clamp(marginals, 1e-7, 1 - 1e-7)
                return torch.logit(marginals_clamped)
            
            return marginals
