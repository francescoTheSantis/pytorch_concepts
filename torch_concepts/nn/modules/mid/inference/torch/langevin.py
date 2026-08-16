"""LangevinDynamics — gradient-based MCMC over the continuous variables of a PGM.

:class:`~torch_concepts.nn.BeliefPropagation` answers the undirected case by
*enumerating* every free variable, so it needs them discrete and finite. A
concept embedding is neither. This engine is the continuous counterpart: it never
enumerates anything, it follows the score.

.. math::

    x^{(t+1)} = x^{(t)} - \\lambda \\nabla_x E(x^{(t)}) + \\sigma\\,\\epsilon,
    \\qquad \\epsilon \\sim \\mathcal{N}(0, I)

With ``noise_scale=None`` the noise is pinned to :math:`\\sigma = \\sqrt{2\\lambda}`
and the chain is Langevin dynamics proper, whose stationary distribution is
:math:`p(x) \\propto e^{-E(x)}`. The default decouples the two, which is what
every practical energy-based model does — see :class:`LangevinDynamics` for why.

References
----------
Welling & Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics",
ICML 2011.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from ...graph.probabilistic_model import ProbabilisticModel
from ...variable import Variable
from ..utils import flatten_event, reshape_value_to_event
from ....outputs import InferenceOutput
from .base import TorchBaseInference


class LangevinDynamics(TorchBaseInference):
    """Sample a PGM's continuous variables by gradient-based MCMC.

    **Free vs clamped.** The two containers say which is which, exactly as they do
    for every other engine — no new vocabulary:

    ===========================  ==========================================
    ``evidence={'z': v}``        **clamped**: held at ``v`` for the whole chain
    ``query={'z': v}``           **free**, chain initialised at ``v``
    ``query=['z']``              **free**, initialised from noise
    ===========================  ==========================================

    Every name addresses a **whole variable**; there is no member addressing. A
    Markov network models each quantity as its own variable — plates exist to share
    a parametrization across members, which an energy factorized over cliques has
    no use for — so the free/clamped split is resolved per variable and nothing
    here needs to know what a plate is.

    Parameters
    ----------
    pgm : ProbabilisticModel
        Any model whose factors implement ``log_potential`` — a
        :class:`~torch_concepts.nn.MarkovNetwork` of energies, a
        :class:`~torch_concepts.nn.BayesianNetwork`.
    steps : int, default 20
        Chain length ``T``. Short chains are the norm for contrastive divergence;
        long ones for drawing an actual sample.
    step_size : float, default 1.0
        The ``λ`` above. Applied to the *clipped* gradient, so it is a step in
        units of ``grad_clip`` rather than of the raw score.
    noise_scale : float or None, default 0.005
        The ``σ`` above. ``None`` pins it to ``sqrt(2 * step_size)``, making the
        chain exact Langevin dynamics. The decoupled default is deliberate: at the
        step sizes a 20-step chain needs, ``sqrt(2λ)`` is so much larger than the
        drift that the chain is indistinguishable from noise. Every short-run EBM
        tunes the two separately; use ``None`` when you want the sampler to be
        provably targeting ``p``, and are willing to pay for the steps.
    grad_clip : float or None, default 0.03
        Clamp each score component to ``[-grad_clip, grad_clip]`` before stepping.
        The single most important stabiliser for short chains (Du & Mordatch): an
        untrained energy has arbitrarily steep regions, and one unclipped step
        through one of them ends the chain in a place no later step recovers from.
        ``None`` disables.

        Clipping also fixes the chain's **reach**: a coordinate moves at most
        ``steps * step_size * grad_clip`` over the whole run. That budget has to
        exceed the distance from where the chain starts to where the model keeps
        its mass, or the sample never arrives however well the energy is trained.
    anneal : float, default 1.0
        Per-step geometric decay of the **noise only**, so the chain explores early
        and refines late. ``1.0`` (the default) holds it constant and reproduces an
        unannealed run exactly. A chain of ``T`` steps ends at ``σ·anneal**T``.

        This is what separates a sampler from an optimiser. At a noise low enough
        to settle inside a mode, a chain cannot cross between modes; at one high
        enough to cross, it cannot settle — so a *fixed* noise gives either
        collapsed point masses or a diffuse cloud, never the target's own spread.
        Decaying from the second regime into the first is the standard fix (Song &
        Ermon's annealed Langevin).

        ``step_size`` deliberately does **not** decay with it. Shrinking the drift
        alongside the noise leaves their ratio fixed, so the chain is no more
        settled at the end than at the start — measured on a 2-mode mixture, that
        gives a within-mode spread of 5.6 against a target of 1.0, whereas decaying
        the noise alone gives 0.8–1.2.

        Needs a reasonably trained energy: on a poor one the wide early phase just
        scatters the chain.

        This is *not* the same thing as
        :class:`~torch_concepts.nn.AnnealedLangevinDynamics`, which walks a ladder
        the energy was trained on. ``anneal`` decays the noise over a **fixed,
        unconditional** energy, and is the only annealing available when the energy
        has no ``sigma`` input at all.
    buffer_size : int, default 0
        Size of the persistent replay buffer (PCD). ``0`` — the default — means
        plain CD: every chain starts from whatever the query supplied. With a
        buffer, chains resume from past samples, which mixes far better across
        training but decorrelates the negatives from the current batch.
    reinit_prob : float, default 0.05
        With a buffer, the fraction of each batch restarted from noise so the
        buffer cannot collapse onto a single mode.

    Notes
    -----
    The returned sample is **detached**. The chain deliberately does not build a
    graph (``create_graph=False``), because contrastive divergence differentiates
    the energy *at* the negative sample, not through the sampler that produced it
    — and backpropagating through ``T`` steps would cost ``T`` times the memory
    for a gradient the objective does not use.
    """

    name = "LangevinDynamics"

    def __init__(
        self,
        pgm: ProbabilisticModel,
        steps: int = 20,
        step_size: float = 1.0,
        noise_scale: Optional[float] = 0.005,
        grad_clip: Optional[float] = 0.03,
        anneal: float = 1.0,
        buffer_size: int = 0,
        reinit_prob: float = 0.05,
        **base_kwargs,
    ):
        super().__init__(pgm, **base_kwargs)
        if steps < 0:
            raise ValueError(f"{self.name}: `steps` must be non-negative, got {steps}.")
        if anneal <= 0:
            raise ValueError(f"{self.name}: `anneal` must be positive, got {anneal}.")
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.noise_scale = None if noise_scale is None else float(noise_scale)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.anneal = float(anneal)
        self.buffer_size = int(buffer_size)
        self.reinit_prob = float(reinit_prob)
        #: Replay buffer, keyed by variable name; empty unless ``buffer_size > 0``.
        self._buffer: Dict[str, torch.Tensor] = {}

    def __repr__(self) -> str:
        return self._format_repr(
            steps=self.steps,
            step_size=self.step_size,
            noise_scale=self.noise_scale,
            grad_clip=self.grad_clip,
            anneal=self.anneal,
            buffer_size=self.buffer_size,
        )

    # ------------------------------------------------------------------ utils
    def _dtype_device(self) -> tuple:
        """Where the chain runs: the energy's own dtype/device, or the defaults."""
        try:
            reference = next(self.pgm.parameters())
        except StopIteration:
            return torch.get_default_dtype(), torch.device("cpu")
        return reference.dtype, reference.device

    def _to_flat(
        self, mapping: Dict[str, Optional[torch.Tensor]], dtype, device
    ) -> Dict[str, torch.Tensor]:
        """``{name: value}`` -> ``{name: (*leading, size)}``, event dims flattened.

        ``None`` values are dropped, which is how a query says "free, no starting
        point". Serves ``evidence`` and ``query`` alike — the two containers differ
        in meaning, not in shape.
        """
        return {
            name: flatten_event(
                self.pgm.variables[name], value.to(dtype=dtype, device=device)
            )
            for name, value in mapping.items()
            if value is not None
        }

    def _free_nodes(self, clamped: Dict[str, torch.Tensor]) -> List[Variable]:
        """Variables that take part in some factor and are not clamped.

        Model order, so the chain is deterministic across queries.
        """
        return [
            var
            for var in self.pgm.variables.values()
            if self.pgm.factor_names_of(var.name) and var.name not in clamped
        ]

    # ------------------------------------------------------------ the chain
    def _initial_state(
        self,
        nodes: List[Variable],
        supplied: Dict[str, torch.Tensor],
        leading: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Where each free member's chain starts: query value, buffer, or noise."""
        state: Dict[str, torch.Tensor] = {}
        rows = int(leading.numel())
        for node in nodes:
            value = supplied.get(node.name)
            if value is not None:
                state[node.name] = value.to(dtype=dtype, device=device).clone()
                continue
            state[node.name] = self._init_scale() * torch.randn(
                *leading, node.size, dtype=dtype, device=device
            )

        if self.buffer_size > 0:
            for node in nodes:
                stored = self._buffer.get(node.name)
                if stored is None or stored.shape[-1] != node.size:
                    continue
                index = torch.randint(0, stored.shape[0], (rows,), device=stored.device)
                drawn = stored[index].to(dtype=dtype, device=device)
                drawn = drawn.reshape(*leading, node.size)
                # Restart a slice of the batch from the fresh init, so the buffer
                # cannot collapse onto whatever mode it found first.
                keep = (torch.rand(*leading, 1, device=device) >= self.reinit_prob)
                state[node.name] = torch.where(keep, drawn, state[node.name])
        return state

    def _push_buffer(self, state: Dict[str, torch.Tensor]) -> None:
        """Append the finished chain to the replay buffer, oldest evicted first."""
        if self.buffer_size <= 0:
            return
        for name, value in state.items():
            flat = value.reshape(-1, value.shape[-1]).detach()
            stored = self._buffer.get(name)
            merged = flat if stored is None else torch.cat([stored, flat], dim=0)
            self._buffer[name] = merged[-self.buffer_size:]

    def _layout(
        self,
        clamped: Dict[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
        anchors: Dict[str, tuple],
        release: Optional[Dict[str, int]] = None,
    ) -> tuple:
        """Pack every member taking part in a factor into ONE tensor.

        Free and clamped members share the layout: a step then costs a fixed
        number of kernels instead of a handful per member, which is what dominates
        at the sizes an embedding model uses (a 2-D embedding over a 2048-row
        batch makes every op launch-bound).

        Variables are laid out in **model order**, so the layout is deterministic
        across queries and each variable's value is a single contiguous slice —
        which is what lets the output be sliced out directly rather than
        concatenated back together.

        Returns
        -------
        tuple
            ``(flat, hold, pinned, release_at, blocks)``. ``blocks`` is
            ``[(variable, span)]`` in layout order, serving both the factors and the
            output. ``hold`` and ``pinned`` are ``None`` when nothing is held, so the
            chain can skip the write-back entirely. ``release_at`` is one integer per
            *column* — the stage at which that coordinate stops being held — and is
            ``None`` unless some variable was given a release point.
        """
        release = release or {}
        # Past the last stage, so an unnamed variable is held for the whole run:
        # "held forever" is just a release point beyond the end of the chain.
        never = self._stage_count() + 1

        columns, pins, holds, releases, blocks = [], [], [], [], []
        offset = 0
        for var in self.pgm.variables.values():
            if not self.pgm.factor_names_of(var.name):
                continue
            held = clamped.get(var.name)
            if held is not None:
                column = pin = held
                hold = torch.ones_like(held, dtype=torch.bool)
                # Evidence is clamped for the whole chain, full stop. Timed release
                # is for values passed through `query`; see :meth:`query`.
                stage_at = never
            else:
                column = state[var.name]
                anchor = anchors.get(var.name)
                # The pin is the *anchor's* value, not the state's: a replay
                # buffer can have overwritten what `supplied` initialised.
                pin = anchor[0] if anchor is not None else column
                stage_at = int(release.get(var.name, never))
                if anchor is not None:
                    hold = anchor[1]
                elif var.name in release:
                    # A release point promotes a query value from "initial point"
                    # to "held until stage k" — the whole feature.
                    hold = torch.ones_like(column, dtype=torch.bool)
                else:
                    hold = torch.zeros_like(column, dtype=torch.bool)
            columns.append(column)
            pins.append(pin)
            holds.append(hold)
            releases.append(torch.full((var.size,), stage_at))
            blocks.append((var, slice(offset, offset + var.size)))
            offset += var.size

        held_anything = bool(clamped or anchors or release)
        return (
            torch.cat(columns, dim=-1),
            torch.cat(holds, dim=-1) if held_anything else None,
            torch.cat(pins, dim=-1) if held_anything else None,
            torch.cat(releases) if release else None,
            blocks,
        )

    def _stage_count(self) -> int:
        """How many stages this chain has — one per step; one per rung when annealed."""
        return self.steps

    def _run_chain(
        self,
        flat: torch.Tensor,
        hold: Optional[torch.Tensor],
        pinned: Optional[torch.Tensor],
        release_at: Optional[torch.Tensor],
        blocks: List[tuple],
    ) -> torch.Tensor:
        """``T`` score-following steps over the packed state, detached throughout.

        One ``autograd.grad`` per step over the *whole* free set — the factor
        energies are summed first, so the graph is built and walked once however
        many cliques a variable belongs to.

        Parameters
        ----------
        flat : torch.Tensor
            ``(*leading, total)`` — the chain's starting position, every member
            packed side by side by :meth:`_layout`.
        hold : torch.Tensor or None
            Boolean, same shape as ``flat``. True wherever a coordinate is held:
            all columns of an evidence-clamped member, and the masked *rows* of a
            per-row anchored one. ``None`` means nothing is held.
        pinned : torch.Tensor or None
            The values ``hold`` restores to.
        release_at : torch.Tensor or None
            One integer per column: the stage at which that coordinate stops being
            held. ``None`` when nothing is released, so the common path skips the
            comparison entirely.
        blocks : list of (Variable, slice)
            Which columns belong to which member, for feeding the factors.

        Returns
        -------
        torch.Tensor
            ``flat`` at step ``T``, detached.

        Notes
        -----
        Holding a coordinate requires suppressing **both** the gradient step and
        the noise, which is what the single ``torch.where`` does. Zeroing only the
        gradient leaves the noise term free to random-walk the evidence by
        ``σ√T``, and every free member coupled to it inherits that error.

        The per-step rates come from :meth:`_step_schedule`, which is the single place
        the plain and annealed chains differ.
        """
        # Bound before the loop so a ``steps=0`` chain still has a stage to compare
        # against below (nothing ran, so everything held is still held).
        stage = 0
        # ``enable_grad`` because the sampler must run under ``torch.no_grad()``
        # and in ``.eval()`` — the chain needs the score even when the caller
        # wants nothing differentiated.
        with torch.enable_grad():
            for stage, step_size, noise, sigma in self._step_schedule():
                free = flat.detach().requires_grad_(True)
                # Slices are views, so rebuilding the dict costs no kernels and
                # the gradient flows back through them into the one parent.
                values = {
                    node.name: reshape_value_to_event(node, free[..., span])
                    for node, span in blocks
                }
                conditioning = {} if sigma is None else {"sigma": sigma}
                grad, = torch.autograd.grad(
                    self.pgm.energy(values, **conditioning).sum(),
                    free,
                    create_graph=False,
                )
                if self.grad_clip is not None:
                    grad = grad.clamp(-self.grad_clip, self.grad_clip)
                flat = free.detach() - step_size * grad
                flat = flat + noise * torch.randn_like(flat)
                if hold is not None:
                    active = hold if release_at is None else hold & (stage < release_at)
                    flat = torch.where(active, self._held_value(pinned, sigma), flat)
        # Report a still-held value exactly as supplied, not at the last rung's noise
        # level: the caller asked to condition on a value, not on a jittered one. A
        # *released* coordinate is left where the chain took it — that is the point.
        if hold is not None:
            final = hold if release_at is None else hold & (stage < release_at)
            flat = torch.where(final, pinned, flat)
        return flat.detach()

    # ------------------------------------------------- hooks a subclass overrides
    def _init_scale(self) -> float:
        """Standard deviation a free variable is drawn from when the query gives none."""
        return 1.0

    @staticmethod
    def _held_value(pinned: torch.Tensor, sigma) -> torch.Tensor:
        """What a held coordinate is restored to *during* the chain: its own value.

        :class:`~torch_concepts.nn.AnnealedLangevinDynamics` overrides this to
        perturb the value to the current rung, which is the inpainting correction a
        noise-conditioned energy needs.
        """
        return pinned

    def _step_schedule(self):
        """``(stage, step_size, noise, sigma)`` per chain step.

        ``steps`` iterations at the configured rate, with the noise decayed by
        ``anneal`` after each. ``sigma`` is ``None`` — this engine's energy takes no
        noise level — and the **stage is the step index**, which is this chain's
        unit of progress and therefore what ``release`` counts in.

        :class:`~torch_concepts.nn.AnnealedLangevinDynamics` overrides this with the
        ladder, where a stage is a rung instead.
        """
        noise = self.noise_scale
        if noise is None:
            noise = (2.0 * self.step_size) ** 0.5
        for stage in range(self.steps):
            yield stage, self.step_size, noise, None
            # Decay after the step, so step 1 always runs at the configured rate
            # and ``anneal=1.0`` is exactly the unannealed chain.
            noise *= self.anneal

    # ------------------------------------------------------------------ query
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        clamp_mask: Optional[Dict[str, torch.Tensor]] = None,
        release: Optional[Dict[str, int]] = None,
        n_samples: Optional[int] = None,
    ) -> InferenceOutput:
        """Run the chain and report the free variables' final state.

        A name in ``evidence`` is clamped; a name in ``query`` is free, and its
        value (if any) is the chain's starting point rather than a constraint.

        Reported in ``out.samples`` only. This engine draws realisations, not
        distribution parameters — there is no posterior ``loc`` to report, and
        calling a Langevin draw one would misname it.

        Parameters
        ----------
        clamp_mask : dict, optional
            **Per-row** clamping, for the case ``evidence`` cannot express: the
            same variable clamped on some rows of the batch and free on others.
            ``clamp_mask[name]`` is a boolean (or 0/1) tensor broadcastable to the
            variable's value; rows where it is true are held at the value the
            *query* supplied for that name, rows where it is false evolve freely.

            This is what a stochastic intervention needs — every sample gets its
            own random subset of intervened concepts, so a whole-variable clamp
            would either freeze the batch or none of it. Names given here must
            appear in ``query`` with a value (the thing being held).
        release : dict, optional
            ``{name: stage}`` — hold a variable's value up to ``stage``, then let it
            evolve. **Repainting**: a coordinate is filled in for part of the run and
            free for the rest, which neither container can say on its own.

            A *stage* is this chain's unit of progress: the **step** index here, the
            **rung** index for
            :class:`~torch_concepts.nn.AnnealedLangevinDynamics`. So
            ``release={'x': 3}`` holds ``x`` through stages 0-2 and frees it at 3.

            **Only names a ``query`` variable carrying a tensor.** That is the one
            case with a meaning: evidence is clamped for the whole chain by
            definition, and a variable the query leaves valueless starts from noise,
            so holding it would pin a random draw. Both are rejected, as is a name
            that appears nowhere.

            So ``release`` extends ``query``'s own contract rather than blurring the
            two containers:

            ============================  ==========================================
            ``query={'x': v}``            free from step 0, ``v`` is just the start
            ``query={'x': v}``, rel ``k`` held at ``v`` to stage ``k``, free after
            ``evidence={'x': v}``         held at ``v`` for the whole chain
            ============================  ==========================================

            A released coordinate keeps whatever value it holds at that moment and
            evolves from there; it is not re-initialised. Unlike a still-held one, it
            is reported as the chain left it, not as it was supplied.
        n_samples : int, optional
            Batch size for a pass where neither the evidence nor the query carries
            a tensor to read one from — ``query=['x', 'y']`` with no evidence.
            Ignored otherwise, since a supplied tensor already fixes the leading
            shape.

            A chain started this way needs no hand-built initial value: an
            un-supplied variable is drawn from ``N(0, I)``, or from
            ``N(0, sigmas[0]**2 I)`` under
            :class:`~torch_concepts.nn.AnnealedLangevinDynamics` — which is exactly
            the start that engine prescribes.
        """
        evidence = dict(evidence or {})
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        query_names = list(query)
        leading = self._query_leading_shape(query, evidence, default=n_samples)
        dtype, device = self._dtype_device()

        clamped = self._to_flat(evidence, dtype, device)
        supplied = self._to_flat(query, dtype, device)
        self._validate_release(release, supplied, clamped)

        nodes = self._free_nodes(clamped)
        if not nodes:
            return InferenceOutput()

        # Query values initialise the chain; they do not constrain it.
        state = self._initial_state(nodes, supplied, leading, dtype, device)
        anchors = self._expand_anchors(clamp_mask, supplied, clamped, device)

        flat, hold, pinned, release_at, blocks = self._layout(
            clamped, state, anchors, release
        )
        flat = self._run_chain(flat, hold, pinned, release_at, blocks)

        spans = {var.name: span for var, span in blocks}
        self._push_buffer({n.name: flat[..., spans[n.name]] for n in nodes})
        return InferenceOutput(
            samples=self._assemble_samples(
                {name: flat[..., span] for name, span in spans.items()}, query_names
            )
        )

    def _validate_release(
        self,
        release: Optional[Dict[str, int]],
        supplied: Dict[str, torch.Tensor],
        clamped: Dict[str, torch.Tensor],
    ) -> None:
        """``release`` may only name a ``query`` variable carrying a tensor.

        The other three cases all had a plausible-looking but meaningless reading,
        so each is rejected rather than guessed at:

        * **evidence** — clamped for the whole chain by definition. Releasing it
          would make the two containers mean the same thing with different defaults;
          pass the value through ``query`` instead.
        * **a query name with no tensor** — the chain starts it from noise, so a
          hold would pin a *random draw* for ``k`` stages. Held nothing, released
          nothing.
        * **a name that is nowhere** — a typo, previously ignored in silence.
        """
        if not release:
            return
        for name in release:
            if name in supplied:
                continue
            if name in clamped:
                raise ValueError(
                    f"{self.name}: release names {name!r}, which is evidence — and "
                    "evidence is clamped for the whole chain. To hold a value and "
                    "then let it go, pass it in `query` instead of `evidence`."
                )
            raise ValueError(
                f"{self.name}: release names {name!r}, but the query supplies no "
                "tensor for it. There is nothing to hold — a released variable is "
                "one whose value you gave, as query={{{name!r}: value}}.".format(
                    name=name
                )
            )

    def _expand_anchors(
        self,
        clamp_mask: Optional[Dict[str, torch.Tensor]],
        supplied: Dict[str, torch.Tensor],
        clamped: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, tuple]:
        """``{name: (held value, boolean mask)}`` from a per-row ``clamp_mask``.

        The held value is what the query supplied — clamping a row to a value the
        caller never gave would be meaningless.
        """
        if not clamp_mask:
            return {}
        anchors: Dict[str, tuple] = {}
        for name, mask in clamp_mask.items():
            value = supplied.get(name)
            if value is None:
                raise ValueError(
                    f"{self.name}: clamp_mask names {name!r}, but the query "
                    f"supplies no value for it to hold it at. A per-row clamp "
                    "pins rows to the value passed in `query`."
                )
            if name in clamped:
                continue  # wholly clamped by evidence already — nothing to pin
            mask = mask.to(device=device, dtype=torch.bool)
            anchors[name] = (value, mask.expand_as(value))
        return anchors
