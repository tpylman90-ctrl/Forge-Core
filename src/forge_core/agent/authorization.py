"""Reference authorization policy; not an authenticated production mechanism.

Real execution MUST bind approvals to trusted user UI events, resources,
identities, TTL, audit logs and revocation. Model text may not grant approval.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionRequest:
    verb: str
    resource: str


@dataclass(frozen=True)
class Approval:
    request: ActionRequest
    user_approved: bool


def require_approval(request: ActionRequest, approval: Approval | None) -> None:
    """Deny unless an explicit affirmative approval matches this action exactly."""
    if approval is None or not approval.user_approved or approval.request != request:
        raise PermissionError(f"External action not authorized: {request.verb}")
