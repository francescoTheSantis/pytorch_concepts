"""
Example: Sampling-based Marginal Inference using Pyro

This example demonstrates how to use SamplingInference to perform approximate marginal
queries on trained concept-based models. We train a simple CBM on a car starting
dataset and then use Pyro-based importance sampling to estimate marginal probabilities.

Scenario: A car starts based on engine and wheels status.
- engine (0=broken, 1=working) and wheels (0=broken, 1=working) determine
- car_start (0=doesn't start, 1=starts)

Key Features:
- Train CBM using DeterministicInference (standard approach)
- Use SamplingInference for marginal queries: p(c_i | x), p(c_i | c_j, x)
- Compare approximate marginals with deterministic predictions
- Demonstrate conditional probability queries

Note: This uses importance sampling (Monte Carlo approximation), not exact inference!
Accuracy improves with more samples: error ~ O(1/√num_samples)

"""

import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, RelaxedOneHotCategorical

from torch_concepts import LatentVariable, ConceptVariable
from torch_concepts.data.datasets import ToyDAGDataset
from torch_concepts.nn import (
    LinearLatentToConcept, 
    LinearConceptToConcept, 
    ParametricCPD, 
    ProbabilisticModel,
    DeterministicInference,
    SamplingInference,
    LazyConstructor
)


def main():

    # ========================================================================
    #  1. We create the synthetic dataset 
    # ========================================================================

    # We use the ToyDAGDataset to create a simple, synthetic dataset corresponding to
    # a DAG with 3 variables: engine, wheels, car_start.
    # As shown in conditional_probs, we also specify the true CPDs. This will be useful for verifying that 
    # our exact inference results align with the known probabilities.

    latent_dims = 16
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    
    dataset = ToyDAGDataset(
        variables=['engine', 'wheels', 'car_start'],
        cardinalities={'engine': 2, 'wheels': 2, 'car_start': 2},
        dag=[('engine', 'car_start'), ('wheels', 'car_start')],
        conditional_probs={
            'car_start': {
                "engine=0,wheels=0": [0.95, 0.05],  # Both broken: 95% won't start
                "engine=0,wheels=1": [0.90, 0.10],  # Engine broken: 90% won't start
                "engine=1,wheels=0": [0.85, 0.15],  # Wheels broken: 85% won't start
                "engine=1,wheels=1": [0.05, 0.95],  # Both working: 95% starts
            }
        },
        seed=42,
        n_gen=n_samples,
        target_variable='car_start',
        autoencoder_kwargs={'latent_dim': latent_dims, 'epochs': 1000}
    )
    
    x_train = dataset.input_data
    # Extract concepts: engine (col 0), wheels (col 1), car_start (col 2)
    c_train = dataset.concepts[:, :2]  # engine and wheels
    y_train = dataset.concepts[:, 2:3]  # car_start
    concept_names = ['engine', 'wheels']
    
    # Convert task to multi-class (for Categorical distribution)
    y_train = torch.cat([1 - y_train, y_train], dim=1)
    
    print(f"Dataset: {n_samples} samples")
    print(f"Input features: {x_train.shape[1]}")
    print(f"Concepts: {concept_names}")
    print(f"Task: Car starting prediction (binary classification)\n")
    
    # Define PGM structure: input -> [engine, wheels] -> car_start
    input_var = LatentVariable("input", parents=[], size=latent_dims)
    concepts = ConceptVariable(concept_names, parents=["input"], distribution=Bernoulli)
    task_var = ConceptVariable("car_start", parents=concept_names, distribution=RelaxedOneHotCategorical, size=2)


    # ========================================================================
    # 2. We train a simple Concept Model 
    # ========================================================================

    # A simple concept-based model is defined using ParametricCPD for the encoders and predictor. 
    # We train this model to approximate p(engine, wheels, car_start | x) using DeterministicInference.

    # Define CPDs (neural networks)
    backbone = ParametricCPD(
        "input", 
        parametrization=torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], latent_dims), 
            torch.nn.LeakyReLU()
        )
    )
    c_encoder = ParametricCPD(
        ["engine", "wheels"], 
        parametrization=LazyConstructor(LinearLatentToConcept)
    )
    y_predictor = ParametricCPD(
        "car_start", 
        parametrization=LinearConceptToConcept(in_concepts=2, out_concepts=2)
    )
    
    # Create ProbabilisticModel
    concept_model = ProbabilisticModel(
        variables=[input_var, *concepts, task_var],
        parametric_cpds=[backbone, *c_encoder, y_predictor]
    )
    
    # Train using DeterministicInference
    inference_engine = DeterministicInference(concept_model)
    initial_input = {'input': x_train}
    query_concepts = ["engine", "wheels", "car_start"]
    
    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        cy_pred = inference_engine.query(query_concepts, evidence=initial_input, debug=True)
        c_pred = cy_pred[:, :c_train.shape[1]]
        y_pred = cy_pred[:, c_train.shape[1]:]
        
        # Compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train.argmax(dim=1), y_pred.argmax(dim=1))
            concept_accuracy = accuracy_score(c_train, c_pred > 0.)
            print(f"Epoch {epoch:3d}: Loss {loss.item():.4f} | "
                  f"Task Acc: {task_accuracy:.3f} | Concept Acc: {concept_accuracy:.3f}")


    """Demonstrate sampling-based marginal inference capabilities."""
    print("=" * 60)
    print("STEP 2: Sampling-based Marginal Inference with Pyro")
    print("=" * 60)
    

    # ========================================================================
    # 3. Marginal inference 
    # ========================================================================
    
    # QUERIES:  p(engine | x), p(wheels | x)
    # By construction (toy dataset), we know the the conditional for a root concept is p(engine | x) = 0.5
    # Therefore, we expect the approximate marginal probabilities for engine to be around 0.5 for all samples.
    # The same applies to p(wheels | x) = 0.5, since it is also a root concept.
    # Note: Results will have some Monte Carlo error that decreases with num_samples.

    # Set model to eval mode
    concept_model.eval()
    
    # Create sampling inference engine
    print("Creating SamplingInference engine with Pyro...")
    sampling_inference = SamplingInference(concept_model, num_samples=1000)
    print("✓ SamplingInference initialized\n")
    print("Algorithm: Importance Sampling (Monte Carlo approximation)")
    print("Approximate error: ~3% with 1000 samples\n")
    
    # Use a smaller subset for demonstration
    n_demo = 10
    x_demo = x_train[:n_demo]

    p_engine_marginal = sampling_inference.marginal(['engine'], evidence={'input': x_demo})
    
if __name__ == "__main__":
    main()




    # # ========================================================================
    # # QUERY 2: Marginal p(wheels | x)
    # # ========================================================================
    # print("-" * 60)
    # print("Query 2: Marginal p(wheels | x)")
    # print("-" * 60)
    
    # p_wheels_marginal = sampling_inference.marginal(['wheels'], evidence={'input': x_demo})
    
    # print("Exact marginal probabilities p(wheels=working | x):")
    # print(p_wheels_marginal.squeeze().numpy())
    
    # wheels_logits = det_inference.query(['wheels'], evidence={'input': x_demo})
    # wheels_probs_det = torch.sigmoid(wheels_logits)
    
    # agreement = (torch.abs(p_wheels_marginal - wheels_probs_det) < 0.1).float().mean()
    # print(f"\nAgreement with deterministic (±0.1): {agreement.item():.1%}\n")
    
    # # ========================================================================
    # # QUERY 3: Joint p(engine, wheels | x)
    # # ========================================================================
    # print("-" * 60)
    # print("Query 3: Joint marginal p(engine, wheels | x)")
    # print("-" * 60)
    
    # p_engine_wheels_joint = sampling_inference.marginal(['engine', 'wheels'], evidence={'input': x_demo})
    
    # print("Joint marginal [p(engine=working | x), p(wheels=working | x)]:")
    # for i in range(min(5, n_demo)):
    #     print(f"Sample {i}: engine={p_engine_wheels_joint[i, 0]:.3f}, wheels={p_engine_wheels_joint[i, 1]:.3f}")
    # print(f"Shape: {p_engine_wheels_joint.shape}\n")
    
    # # ========================================================================
    # # QUERY 4: Conditional p(wheels | engine=working, x)
    # # ========================================================================
    # print("-" * 60)
    # print("Query 4: Conditional p(wheels | engine=working, x)")
    # print("-" * 60)
    
    # # Fix engine = 1 (working) for all samples
    # engine_observed = torch.ones(n_demo, 1)
    # p_wheels_given_engine_working = sampling_inference.marginal(
    #     ['wheels'], 
    #     evidence={'input': x_demo, 'engine': engine_observed}
    # )
    
    # print("Conditional probabilities p(wheels=working | engine=working, x):")
    # print(p_wheels_given_engine_working.squeeze().numpy())
    
    # # Compare with when engine is not fixed
    # print("\nCompare with marginal p(wheels | x):")
    # print(p_wheels_marginal.squeeze().numpy())
    
    # # Show the effect of conditioning
    # print("\nEffect of conditioning on engine=working:")
    # for i in range(min(5, n_demo)):
    #     diff = p_wheels_given_engine_working[i, 0].item() - p_wheels_marginal[i, 0].item()
    #     print(f"Sample {i}: Δp = {diff:+.3f}")
    # print()
    
    # # ========================================================================
    # # QUERY 5: Conditional p(wheels | engine=broken, x)
    # # ========================================================================
    # print("-" * 60)
    # print("Query 5: Conditional p(wheels | engine=broken, x)")
    # print("-" * 60)
    
    # # Fix engine = 0 (broken) for all samples
    # engine_observed_zero = torch.zeros(n_demo, 1)
    # p_wheels_given_engine_broken = sampling_inference.marginal(
    #     ['wheels'], 
    #     evidence={'input': x_demo, 'engine': engine_observed_zero}
    # )
    
    # print("Conditional probabilities p(wheels=working | engine=broken, x):")
    # print(p_wheels_given_engine_broken.squeeze().numpy())
    
    # print("\nCompare p(wheels|engine=broken) vs p(wheels|engine=working):")
    # for i in range(min(5, n_demo)):
    #     p_broken = p_wheels_given_engine_broken[i, 0].item()
    #     p_working = p_wheels_given_engine_working[i, 0].item()
    #     print(f"Sample {i}: p(w|e=broken)={p_broken:.3f}, p(w|e=working)={p_working:.3f}, Δ={p_working-p_broken:+.3f}")
    # print()
    
    # # ========================================================================
    # # QUERY 6: Task prediction p(car_start | x)
    # # ========================================================================
    # print("-" * 60)
    # print("Query 6: Task marginal p(car_start | x)")
    # print("-" * 60)
    
    # p_task_marginal = sampling_inference.marginal(['car_start'], evidence={'input': x_demo})
    
    # print("Task probabilities [p(doesn't start), p(starts)]:")
    # for i in range(min(5, n_demo)):
    #     print(f"Sample {i}: [{p_task_marginal[i, 0]:.3f}, {p_task_marginal[i, 1]:.3f}] "
    #           f"-> prediction: {'starts' if p_task_marginal[i].argmax().item() == 1 else 'doesn\'t start'}")
    
    # # Compare with deterministic
    # task_logits = det_inference.query(['car_start'], evidence={'input': x_demo})
    # task_probs_det = torch.softmax(task_logits, dim=-1)
    
    # print("\nDeterministic task probabilities:")
    # for i in range(min(5, n_demo)):
    #     print(f"Sample {i}: [{task_probs_det[i, 0]:.3f}, {task_probs_det[i, 1]:.3f}] "
    #           f"-> prediction: {'starts' if task_probs_det[i].argmax().item() == 1 else 'doesn\'t start'}")
    
    # # Compute classification agreement
    # exact_preds = p_task_marginal.argmax(dim=1)
    # det_preds = task_probs_det.argmax(dim=1)
    # pred_agreement = (exact_preds == det_preds).float().mean()
    # print(f"\nPrediction agreement: {pred_agreement.item():.1%}\n")
    
    # # ========================================================================
    # # QUERY 7: Task prediction p(car_start | engine=working, wheels=broken, x)
    # # ========================================================================
    # print("-" * 60)
    # print("Query 7: Task conditional p(car_start | engine=working, wheels=broken, x)")
    # print("-" * 60)
    
    # engine_obs = torch.ones(n_demo, 1)
    # wheels_obs = torch.zeros(n_demo, 1)
    
    # p_task_conditional = sampling_inference.marginal(
    #     ['car_start'], 
    #     evidence={'input': x_demo, 'engine': engine_obs, 'wheels': wheels_obs}
    # )
    
    # print("Task probabilities p(car_start | engine=working, wheels=broken, x):")
    # for i in range(min(5, n_demo)):
    #     print(f"Sample {i}: [{p_task_conditional[i, 0]:.3f}, {p_task_conditional[i, 1]:.3f}] "
    #           f"-> prediction: {'starts' if p_task_conditional[i].argmax().item() == 1 else 'doesn\'t start'}")
    
    # print("\nWith engine=working but wheels=broken, car should mostly not start (CPT: 85% won't start)")
    # # Predict class 0 (doesn't start)
    # correct_preds = (p_task_conditional.argmax(dim=1) == 0).float().mean()
    # print(f"Correct predictions (doesn't start): {correct_preds.item():.1%}\n")
