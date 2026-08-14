"""CFGSamplingEngine — reverse-diffusion sampling with classifier-free guidance."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from ...utils import flatten_event, reshape_value_to_event
from .....outputs import InferenceOutput
from ..ancestral import AncestralSamplingInference
from .schedule import DiffusionSchedule


class CFGSamplingEngine(AncestralSamplingInference):
    """Draw from a diffusion model by running its reverse process.

    Every other engine in the library answers a query with **one** pass over the
    graph. This one cannot: a diffusion model's generative distribution is not
    the training graph, it is that graph's denoising CPD applied ``n_steps``
    times in sequence. The loop has to live somewhere, and putting it behind the
    standard :meth:`query` signature is what lets the rest of the stack — the
    FID and steerability metrics above all — treat this model like any other
    generative one.

    What a query means here
    -----------------------
    ``query`` names the variables to report; ``evidence`` decides what is fixed
    rather than drawn:

    * **concepts in the evidence** — conditional generation. This is also how an
      intervention arrives, so ``do(concept)`` works unchanged.
    * **concepts absent** — they are drawn from the model's learned marginal
      ``p(c)`` by an ordinary ancestral pass, which is what an *unconditional*
      draw (for FID) needs.
    * **the noise variable in the evidence** — the reverse process starts from
      that value instead of a fresh ``N(0, I)``. This is what makes a
      steerability comparison mean anything: the same starting point, one
      concept changed, everything else held.

    Guidance
    --------
    Both arms are evaluated in a **single** batched call — the conditioning is
    stacked ``[y, null]`` along the batch axis and the network runs once on
    ``2B`` rows, which is materially faster than two calls on ``B`` — and
    combined as

    .. math:: \\tilde\\epsilon = (1 + s)\\,\\epsilon_\\theta(x, t, y)
                              - s\\,\\epsilon_\\theta(x, t, \\varnothing)

    ``s`` is :attr:`guidance_scale`; ``s = 0`` is plain conditional sampling.
    The other common convention writes this as
    ``(1-w)*eps_null + w*eps_cond`` with ``w = 1 + s``.

    The unconditional arm is produced by clamping the graph's own mask node to
    zero and re-running the ``y`` CPD, rather than by zeroing the condition here
    — so whatever the model means by "no condition", the sampler inherits it.

    Parameters
    ----------
    pgm : BayesianNetwork
        The diffusion model's graph.
    schedule : DiffusionSchedule
        **The same instance the model's** ``x_t`` **CPD holds.** A sampler running
        a different schedule from the one the network trained against produces
        noise, and no shape or loss would report it.
    guidance_scale : float, default 3.0
        ``s`` above. Higher trades diversity for fidelity to the condition.
    n_steps : int, optional
        Reverse steps. Defaults to the schedule's full ``T``. Fewer strides the
        grid — valid because the DDIM update is defined between any two noise
        levels — which is how FID over thousands of samples stays affordable.
    eta : float, default 0.0
        Reverse-process stochasticity. ``0`` is the deterministic DDIM sampler;
        ``1`` on the full grid is exactly DDPM's Algorithm 2. Left at ``0`` by
        default because injecting fresh noise at every step swamps the shared
        starting point that the steerability metric replays.
    clip : tuple of float, optional
        Range to clamp the predicted clean image into at each step, e.g.
        ``(0., 1.)``. See :meth:`DiffusionSchedule.ddim_step`.
    observation, noise, state, timestep, mask, condition, prediction : str
        Variable names in the graph. The defaults match
        :class:`~torch_concepts.nn.ClassifierFreeGuidedDiffusion`.
    **kwargs
        Forwarded to :class:`~torch_concepts.nn.AncestralSamplingInference`,
        which draws the concepts from ``p(c)``.

    See Also
    --------
    torch_concepts.nn.CFGTrainingEngine : the train-time counterpart
    """

    name = "CFGSamplingEngine"
    is_stochastic = True

    def __init__(
        self,
        pgm,
        schedule: DiffusionSchedule,
        guidance_scale: float = 3.0,
        n_steps: Optional[int] = None,
        eta: float = 0.0,
        clip: Optional[Tuple[float, float]] = None,
        observation: str = "input",
        noise: str = "eps",
        state: str = "x_t",
        timestep: str = "t",
        mask: str = "m",
        condition: str = "y",
        prediction: str = "eps_hat",
        **kwargs,
    ) -> None:
        super().__init__(pgm, **kwargs)
        self.schedule = schedule
        self.guidance_scale = float(guidance_scale)
        self.n_steps = schedule.n_steps if n_steps is None else int(n_steps)
        self.eta = float(eta)
        self.clip = tuple(clip) if clip is not None else None
        self.observation = observation
        self.noise = noise
        self.state = state
        self.timestep = timestep
        self.mask = mask
        self.condition = condition
        self.prediction = prediction

        missing = [
            n for n in (observation, noise, state, timestep, mask, condition, prediction)
            if n not in pgm.variables
        ]
        if missing:
            raise ValueError(
                f"CFGSamplingEngine: {sorted(missing)} not in the model's graph "
                f"(it has {sorted(pgm.variables)}). Pass the right names, or use "
                "this engine with a diffusion model."
            )

    def __repr__(self) -> str:
        return self._format_repr(
            guidance_scale=self.guidance_scale,
            n_steps=self.n_steps,
            eta=self.eta,
            clip=self.clip,
        )

    # ------------------------------------------------------------------
    # Pieces of one reverse step
    # ------------------------------------------------------------------
    @property
    def _concept_names(self) -> List[str]:
        """The graph's concept variables — the condition, in graph order."""
        return [
            v.name for v in self.pgm.variables.values()
            if v.variable_type == "concept"
        ]

    def _conditions(
        self, concepts: Dict[str, torch.Tensor], leading: torch.Size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The conditional and unconditional ``y``, via the graph's own ``y`` CPD.

        Running the CPD with the mask clamped to one and to zero — rather than
        building ``[c, 0]`` here — keeps a single definition of the null token.
        Whatever the model does to an unconditioned row, the sampler does too.
        """
        cpd = self.pgm.factors[self.condition]
        mask_variable = self.pgm.variables[self.mask]

        def run(fill: float) -> torch.Tensor:
            mask = torch.full(
                (*leading, mask_variable.size), fill,
                device=self._device, dtype=self._dtype,
            )
            params = cpd(parent_values={**concepts, self.mask: mask})
            return flatten_event(
                self.pgm.variables[self.condition],
                next(iter(params.values())),
            )

        return run(1.0), run(0.0)

    def _guided_noise(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        y_cond: torch.Tensor,
        y_null: torch.Tensor,
    ) -> torch.Tensor:
        """``eps_hat`` for both guidance arms in one batched call, then combined."""
        cpd = self.pgm.factors[self.prediction]
        double = {
            self.state: torch.cat([x, x], dim=0),
            self.timestep: torch.cat([index, index], dim=0).to(self._dtype),
            self.condition: torch.cat([y_cond, y_null], dim=0),
        }
        params = cpd(parent_values=double)
        stacked = flatten_event(
            self.pgm.variables[self.prediction], next(iter(params.values()))
        )
        eps_cond, eps_null = stacked.chunk(2, dim=0)
        s = self.guidance_scale
        return (1.0 + s) * eps_cond - s * eps_null

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    @torch.no_grad()
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
        layer_kwargs: Optional[Dict[str, Dict]] = None,
        n_samples: Optional[int] = None,
    ) -> InferenceOutput:
        """Generate by running the reverse process, and report the result.

        Gradients are off throughout: this engine is for generation, and a
        ``n_steps``-deep graph over a U-Net would not fit in memory anyway.
        """
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        evidence, _ = self._split_evidence(evidence)
        leading = self._query_leading_shape(query, evidence, default=n_samples)
        if len(leading) != 1:
            # The reverse process carries one state tensor across the loop and
            # doubles it for the two guidance arms, both of which assume a single
            # batch axis. Say so rather than fail later on a shape that has been
            # silently folded.
            raise ValueError(
                f"{self.name}: expected one leading (batch) dimension, got "
                f"{tuple(leading)}. Flatten the extra dimensions before sampling."
            )

        # --- the condition -------------------------------------------------
        # A concept supplied as evidence is used as given (conditional
        # generation, and how an intervention arrives); the rest are drawn from
        # the learned marginal p(c) by an ordinary ancestral pass, which is what
        # an unconditional draw needs.
        #
        # The two cases have to be separated because a fully-observed variable is
        # clamped and its CPD skipped, so it is reported in neither `params` nor
        # `samples` — reading the condition off a sub-query's output alone would
        # find nothing exactly when every concept was supplied. A *partially*
        # observed plate is not evidence for the whole variable, so it correctly
        # falls to the draw below, which applies the member evidence itself.
        concepts: Dict[str, torch.Tensor] = {
            name: reshape_value_to_event(
                self.pgm.variables[name],
                self._format_evidence(self.pgm.variables[name], evidence[name]),
            )
            for name in self._concept_names
            if name in evidence
        }
        missing = [n for n in self._concept_names if n not in concepts]
        if missing:
            drawn = super().query(
                query=missing, evidence=evidence, n_samples=int(leading.numel()),
            )
            for name in missing:
                value = drawn.samples[name]
                concepts[name] = reshape_value_to_event(
                    self.pgm.variables[name], getattr(value, "tensor", value)
                )
        y_cond, y_null = self._conditions(concepts, leading)

        # --- the starting point --------------------------------------------
        observation = self.pgm.variables[self.observation]
        if self.noise in evidence:
            x = flatten_event(
                self.pgm.variables[self.noise],
                self._format_evidence(self.pgm.variables[self.noise], evidence[self.noise]),
            )
        else:
            x = torch.randn(
                *leading, observation.size, device=self._device, dtype=self._dtype
            )
        noise_start = x

        # --- the reverse process --------------------------------------------
        ones = torch.ones(*leading, 1, device=self._device, dtype=torch.long)
        for step, step_prev in self.schedule.timestep_pairs(self.n_steps):
            index, index_prev = ones * step, ones * step_prev
            eps = self._guided_noise(x, index, y_cond, y_null)
            x = self.schedule.ddim_step(
                x, eps, index, index_prev, eta=self.eta, clip=self.clip
            )

        # --- report -----------------------------------------------------------
        # `input` is the generated image; the concepts and the starting noise go
        # out too, because the steerability metric replays exactly those.
        realised = {
            self.observation: reshape_value_to_event(observation, x),
            self.noise: noise_start,
            **concepts,
        }
        reported = [name for name in query if name in realised]
        params = {
            name: {self._point_estimate_name(name): flatten_event(
                self.pgm.variables[name], realised[name]
            )}
            for name in reported
        }
        return InferenceOutput(
            params=self._assemble_params(params, reported),
            samples=self._assemble_samples(
                {name: realised[name] for name in reported}, reported
            ),
        )

    def _point_estimate_name(self, name: str) -> str:
        """The quantity a realised value is reported under, for this variable.

        A generated value *is* the family's point estimate, so it goes out under
        the name that family's consumers look for — ``value`` for a ``Delta``
        observation, ``probs`` for a Bernoulli concept — which is what
        ``observation_of`` and the concept metrics read.
        """
        from ....distributions import spec_for

        return spec_for(self.pgm.variables[name].distribution).primary_param

    # ------------------------------------------------------------------
    @property
    def _device(self) -> torch.device:
        return self.schedule.alpha_bar.device

    @property
    def _dtype(self) -> torch.dtype:
        return self.schedule.alpha_bar.dtype
