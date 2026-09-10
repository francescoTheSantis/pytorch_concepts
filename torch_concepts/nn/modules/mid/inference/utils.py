"""Distribution utilities shared across all inference backends.

Backend-agnostic helpers used by both the pure-PyTorch and the Pyro engines:
temperature schedules, event reshaping, exact distribution construction,
teacher forcing, and the discrete-state count an enumeration-based engine needs.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dist

from ..distributions import spec_for
from ..variable import Variable


# ---------------------------------------------------------------------------
# Teacher forcing
# ---------------------------------------------------------------------------
# Shared by both backends: ``ForwardInference`` applies it to the value it
# propagates, ``PyroBaseInference.model_fn`` to the value a sample site caches.
# See ``BaseInference``'s ``p_int`` for what the rate means.

def _align_gt(
    gt: torch.Tensor, ref: torch.Tensor, name: Optional[str] = None
) -> torch.Tensor:
    """Cast and reshape ground-truth tensor to match the dtype and shape of ref.

    Step-by-step:
    1. Cast ``gt`` to ``ref``'s dtype so arithmetic ops don't raise type errors
       (e.g. LongTensor label vs FloatTensor network output).
    2. If shapes already match after the cast, return immediately.
    3. Handle the common "extra trailing 1" mismatches that arise when some
       code paths squeeze scalars and others don't:
       - ``gt`` has one more dim than ``ref`` and its last dim is 1 → squeeze it off.
       - ``gt`` has one fewer dim than ``ref`` and ``ref``'s last dim is 1 → unsqueeze.
    4. Finally, broadcast ``gt`` to exactly ``ref``'s shape so downstream ops
       (e.g. per-element masking) can use ``gt`` in place of ``ref``.

    Step 4 also silently rescues genuinely wrong targets — e.g. a ``(B,)`` label
    stretched across a ``(B, k)`` output — so a broadcast that is not one of the
    trailing-1 cases warns once per (name, shape) pair.
    """
    aligned = gt.to(ref.dtype) if gt.dtype != ref.dtype else gt
    if aligned.shape == ref.shape:
        return aligned

    original_shape = tuple(aligned.shape)
    if aligned.dim() == ref.dim() + 1 and aligned.shape[-1] == 1:
        aligned = aligned.squeeze(-1)
    elif aligned.dim() + 1 == ref.dim() and ref.shape[-1] == 1:
        aligned = aligned.unsqueeze(-1)
    if aligned.shape == ref.shape:
        return aligned

    _warn_broadcast(name, original_shape, tuple(ref.shape))
    return aligned.expand_as(ref)


# (name, target shape, reference shape) triples already warned about, so a
# training loop reports a suspicious target once rather than every step.
_BROADCAST_WARNED: set = set()


def _warn_broadcast(name: Optional[str], gt_shape: tuple, ref_shape: tuple) -> None:
    key = (name, gt_shape, ref_shape)
    if key in _BROADCAST_WARNED:
        return
    _BROADCAST_WARNED.add(key)
    target = f"for {name!r}" if name else "for a query variable"
    warnings.warn(
        f"Teacher forcing {target}: the target of shape {gt_shape} does not match "
        f"the predicted shape {ref_shape} and is being broadcast to fit. This is "
        "usually a mis-shaped label (e.g. class indices where a one-hot or "
        "per-element target is expected); pass a target of shape "
        f"{ref_shape} to silence this.",
        UserWarning,
        stacklevel=3,
    )


def teacher_force(
    nn_value: torch.Tensor,
    gt: torch.Tensor,
    p_int: float,
    n_leading: int,
    name: Optional[str] = None,
) -> torch.Tensor:
    """Stochastically replace nn_value with ground truth at rate p_int.

    The draw is per leading (batch-like) element: ``n_leading`` says how many of
    ``nn_value``'s dimensions are leading, so a variable is forced or not as a
    whole, whatever its event shape and however many batch axes there are.

    A rate strictly between 0 and 1 is CEM's **RandInt**; see
    :class:`~torch_concepts.nn.modules.mid.inference.base.BaseInference`'s
    ``p_int``.
    """
    aligned = _align_gt(gt, nn_value, name)
    if p_int >= 1.0:
        return aligned
    if p_int <= 0.0:
        return nn_value
    mask_shape = nn_value.shape[:n_leading] + (1,) * (nn_value.dim() - n_leading)
    mask = (torch.rand(mask_shape, device=nn_value.device) < p_int).to(nn_value.dtype)
    return mask * aligned + (1.0 - mask) * nn_value


def enumerable_cardinality(variable: Variable) -> int:
    """Number of discrete states of ``variable``.

    Used by the enumeration-based engines (currently
    :class:`~torch_concepts.nn.BeliefPropagation`) to size the state axis of a
    variable's messages and to build a factor's log-potential table.

    - Bernoulli-family with ``size == 1`` -> ``2`` (states ``0`` and ``1``).
    - Categorical/OneHot-family -> ``variable.size`` (one state per class).

    The per-family answer comes from ``DistributionSpec.state_count``.

    Parameters
    ----------
    variable : Variable
        The variable whose discrete states are being counted.

    Returns
    -------
    int
        The number of states.

    Raises
    ------
    ValueError
        If the variable cannot be enumerated — either a ``size > 1``
        Bernoulli (a set of independent bits, not one variable) or a
        non-discrete family such as ``Normal`` or ``Delta``.
    """
    D = variable.distribution
    spec = spec_for(D, f"Variable {variable.name!r}")
    if spec.is_enumerable:
        card = spec.state_count(variable.size)
        if card is not None:
            return card
        raise ValueError(
            f"Variable {variable.name!r}: a size>1 {D.__name__} is a set of "
            "independent bits, not a single enumerable variable. Model each bit "
            "as its own binary variable, or use a Categorical/OneHotCategorical."
        )
    raise ValueError(
        f"Variable {variable.name!r}: distribution {D.__name__} is not discretely "
        "enumerable, so it cannot be a free (queried/latent) variable under belief "
        "propagation. Observe it as evidence, or use a discrete distribution."
    )


def make_temperature_schedule(
    initial_temperature: float,
    annealing: Union[str, Callable[[int], float]],
    annealing_rate: float,
    final_temperature: float = 1e-6,
) -> Callable[[int], float]:
    """Build a ``step -> temperature`` schedule.

    ``annealing`` may be ``'constant'``, ``'exponential'`` (decays as
    ``T0 * exp(-rate * step)``), ``'linear'`` (decays as ``T0 - rate * step``),
    or a user-supplied callable.

    Both decays are clamped below at ``final_temperature``, so the schedule
    reaches a floor and stays there rather than sliding towards zero: a relaxed
    sample is only useful while its gradient is, and a temperature that keeps
    shrinking eventually makes every draw a one-hot with a vanishing gradient.
    Around ``0.1`` a Concrete draw already puts ~94% of its mass on one state.
    A user-supplied callable is returned untouched and is responsible for its
    own floor.
    """
    if callable(annealing):
        return annealing
    floor = float(final_temperature)
    if annealing == "constant":
        return lambda step: float(initial_temperature)
    if annealing == "exponential":
        return lambda step: max(
            floor, float(initial_temperature) * math.exp(-annealing_rate * step)
        )
    if annealing == "linear":
        return lambda step: max(
            floor, float(initial_temperature) - annealing_rate * step
        )
    raise ValueError(
        f"Unknown annealing schedule {annealing!r}. Use "
        "'constant', 'exponential', 'linear', or pass a callable."
    )


#: Hard counterpart of each relaxed family. The estimators draw *exact*
#: samples so that equality matching works, even from a variable declared with
#: a Concrete/relaxed family for gradient flow.
EXACT_FAMILY: Dict[type, type] = {
    dist.RelaxedBernoulli: dist.Bernoulli,
    dist.RelaxedOneHotCategorical: dist.OneHotCategorical,
}

# The straight-through families are looked up by exact key too, so they need
# their own entries — a subclass of RelaxedBernoulli does not match it here.
try:
    import pyro.distributions as _pyro_dist
except ImportError:  # pragma: no cover - pyro not installed
    pass
else:
    EXACT_FAMILY[_pyro_dist.RelaxedBernoulliStraightThrough] = dist.Bernoulli
    EXACT_FAMILY[_pyro_dist.RelaxedOneHotCategoricalStraightThrough] = dist.OneHotCategorical


def build_in_member_layout(
    variable: Variable,
    spec,
    params: Dict[str, torch.Tensor],
    make: Callable[[Dict[str, torch.Tensor]], dist.Distribution],
) -> dist.Distribution:
    """Build ``make(params)`` over the canonical member layout.

    Parameters are reshaped to ``(*leading, n_members, *member_shape)``, so the
    family sees **one member per batch row**: ``k`` members of ``m`` classes are
    ``k`` independent categoricals rather than one distribution over ``k*m``
    classes, and that now falls out of the layout instead of needing a fold,
    build, wrap and reshape-back round trip.

    The member axis and the member's own event are then reinterpreted as the
    event, leaving ``batch_shape == (*leading,)`` — what a ``pyro.plate`` over
    the batch needs, and what makes ``log_prob`` return one score per leading
    element. ``spec.event_ndims`` is subtracted because a family such as
    ``OneHotCategorical`` already claims its trailing class axis as its event.
    """
    member = {name: variable.to_member(t, name) for name, t in params.items()}
    return dist.Independent(
        make(member), 1 + len(variable.member_shape) - spec.event_ndims
    )


def build_distribution(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    family: Optional[type] = None,
) -> dist.Distribution:
    """Build the exact distribution declared by ``variable``.

    ``family`` overrides which distribution class is built — how an estimator
    asks for a *hard* draw from a variable declared with a relaxed family (see
    :data:`EXACT_FAMILY`). ``variable.dist_kwargs`` (a Concrete family's
    temperature) belongs to the declared family, so it is dropped when the
    family is overridden.

    The result always has ``batch_shape == (*leading,)`` and
    ``event_shape == (n_members, *member_shape)``, so ``log_prob`` returns one
    score per leading element and a draw comes back in the member layout.
    """
    D = family if family is not None else variable.distribution
    dist_kwargs = {} if family is not None else variable.dist_kwargs
    spec = spec_for(D, f"Variable {variable.name!r}")
    return build_in_member_layout(
        variable, spec, params, lambda p: D(**p, **dist_kwargs)
    )
