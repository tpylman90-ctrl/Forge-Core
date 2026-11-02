"""Prototype for the Observe -> Ask Why -> Confirm learning loop.

This module does not observe the phone or issue notifications. A future
permission-controlled Android observer must provide trustworthy context.
No candidate becomes trusted without explicit user confirmation.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PrincipleCandidate:
    observation: str
    why_question: str
    proposed_principle: str
    confirmed: bool = False


@dataclass(frozen=True)
class CuriositySettings:
    """Early-learning default deliberately favors asking useful questions.

    high: ask when an authorized observation is novel OR motivation is unclear.
    balanced: ask when both conditions hold.
    quiet/paused: do not interrupt. Queued follow-ups are future work.
    """
    mode: str = "high"

    def __post_init__(self) -> None:
        if self.mode not in {"high", "balanced", "quiet", "paused"}:
            raise ValueError("mode must be high, balanced, quiet, or paused")


@dataclass(frozen=True)
class ObservationSignal:
    """Facts from a future permission-gated observer, not model guesses."""
    observation_authorized: bool
    novel: bool = False
    motivation_unclear: bool = False
    sensitive_context: bool = False
    unsafe_to_interrupt: bool = False
    already_asked: bool = False


class CuriosityEngine:
    def __init__(self, settings: CuriositySettings | None = None) -> None:
        self.settings = settings or CuriositySettings()

    def should_ask(self, signal: ObservationSignal) -> bool:
        """Reference policy, not a notification scheduler or privacy boundary.

        No arbitrary daily cap in high mode; repeated, private, unauthorized
        and unsafe interruptions are always suppressed.
        """
        if (not signal.observation_authorized or signal.sensitive_context
                or signal.unsafe_to_interrupt or signal.already_asked):
            return False
        if self.settings.mode == "high":
            return signal.novel or signal.motivation_unclear
        if self.settings.mode == "balanced":
            return signal.novel and signal.motivation_unclear
        return False

    def propose(self, observation: str, why_question: str, principle: str) -> PrincipleCandidate:
        if not all((observation.strip(), why_question.strip(), principle.strip())):
            raise ValueError("Observation, question and proposed principle are required")
        return PrincipleCandidate(observation, why_question, principle)

    def confirm(self, candidate: PrincipleCandidate, user_confirmed: bool) -> PrincipleCandidate:
        if not user_confirmed:
            return candidate
        return replace(candidate, confirmed=True)
