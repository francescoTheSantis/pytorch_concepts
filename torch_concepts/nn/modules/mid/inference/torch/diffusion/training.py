"""CFGTrainingEngine — the forward pass of a classifier-free guided diffusion model."""
from __future__ import annotations

from ..ancestral import AncestralSamplingInference


class CFGTrainingEngine(AncestralSamplingInference):
    """Training-time engine for a classifier-free guided diffusion model.

    A pass-through over :class:`~torch_concepts.nn.AncestralSamplingInference`,
    and deliberately so. Everything a CFG denoising step needs is expressed in
    the model's graph rather than in an engine:

    ==========================  =============================================
    training step               where it lives
    ==========================  =============================================
    ``t ~ U{0..T-1}``           the ``t`` root, a ``Uniform`` CPD
    ``eps ~ N(0, I)``           the ``eps`` root, a ``Normal`` CPD
    ``x_t = a_t*x0 + b_t*eps``  the ``x_t`` CPD
    drop the label w.p. ``p``   the ``m`` root and the ``y = m * c`` CPD
    the regression target       the ``eps_target`` CPD
    ==========================  =============================================

    so what remains — walk the DAG in topological order and draw each unobserved
    variable from its own distribution — is exactly ancestral sampling. Putting
    the label drop in the graph rather than here is what buys that: the sampler
    then produces its unconditional arm by *clamping the same node*, instead of
    reimplementing the null token a second time.

    The class exists as the named seam for the one thing that could still need
    engine-level handling. A variable declared ``Bernoulli`` is drawn from its
    **relaxed** surrogate (see
    :func:`~torch_concepts.nn.modules.mid.inference.torch.utils.sample_from`), so
    ``m`` would come out soft and ``y = m * c`` would be a *blend* of the real
    and null conditions rather than a drop — which trains and converges and is
    not classifier-free guidance. The model avoids that by declaring ``m`` with a
    straight-through family; a deployment without Pyro would instead override
    :meth:`_resolve` here to threshold that one variable.

    Parameters
    ----------
    pgm : BayesianNetwork
        The diffusion model's graph.
    **kwargs
        Forwarded to :class:`~torch_concepts.nn.AncestralSamplingInference`.

    See Also
    --------
    torch_concepts.nn.CFGSamplingEngine : the eval-time counterpart
    torch_concepts.nn.ClassifierFreeGuidedDiffusion : the model wiring both
    """

    name = "CFGTrainingEngine"
