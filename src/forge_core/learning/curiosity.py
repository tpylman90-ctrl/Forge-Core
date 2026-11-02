"""An in-memory, non-persistent curiosity loop prototype.

Suggestions are not trusted beliefs until confirmed by a user.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PrincipleCandidate:
    observation: str
    why_question: str
    proposed_principle: str
    confirmed: bool = False


class CuriosityEngine:
    def propose(self, observation: str, why_question: str, principle: str) -> PrincipleCandidate:
        if not all((observation.strip(), why_question.strip(), principle.strip())):
            raise ValueError("Observation, question and proposed principle are required")
        return PrincipleCandidate(observation, why_question, principle)

    def confirm(self, candidate: PrincipleCandidate, user_confirmed: bool) -> PrincipleCandidate:
        if not user_confirmed:
            return candidate
        return replace(candidate, confirmed=True)
