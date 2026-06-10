"""FLAG catalog — single source of truth for outcome / sentiment FLAG
topic tags. CoMeT-owned because the compactor (LLM-driven memory node
author) needs the catalog as part of its structured output schema.

``KindFlag`` covers outcome / sentiment markers. The compactor judges
a subset (USER_FEEDBACK / USER_REJECT / SUCCESS / COMPLETE) at creation
time based on the turn's actual content; the rest (SKILL / USE_SKILL /
PASSIVE) are rule-based and applied outside the compactor.

``CompactorJudgedFlag`` is the explicit subset the compactor's
structured output is constrained to. Rule-based flags are attached
by the caller post-compaction.

Action FLAGs (``FLAG:ACT_*``) are deterministically derived from the
turn's tool set or ingest source and live in the caller's tag namespace
(e.g. ``backend.services.tag_namespace.ActFlag``). The LLM must not
emit them — the compactor schema rejects any FLAG: value not in
``CompactorJudgedFlag``.
"""
from __future__ import annotations

from enum import Enum


FLAG_PREFIX = 'FLAG:'


class KindFlag(str, Enum):
    """Kind markers — skill, system-passive, user feedback, outcome.
    A single node can carry several (e.g. USE_SKILL turn whose
    downstream verification passed is both USE_SKILL and SUCCESS).

    Two attachment paths:

      - **Rule-based** (caller-attached, deterministic): SKILL,
        USE_SKILL, PASSIVE, WORKFLOW.
      - **Compactor-judged** (LLM-attached at compaction time):
        USER_FEEDBACK, USER_REJECT, SUCCESS, COMPLETE — see
        ``CompactorJudgedFlag``.
    """
    SKILL = 'FLAG:SKILL'
    USE_SKILL = 'FLAG:USE_SKILL'
    PASSIVE = 'FLAG:PASSIVE'
    USER_FEEDBACK = 'FLAG:USER_FEEDBACK'
    USER_REJECT = 'FLAG:USER_REJECT'
    SUCCESS = 'FLAG:SUCCESS'
    COMPLETE = 'FLAG:COMPLETE'
    WORKFLOW = 'FLAG:WORKFLOW'  # rule-based: a saved workflow template's memory node (mirror of SKILL)


class CompactorJudgedFlag(str, Enum):
    """KindFlag subset the compactor is allowed to emit.

    Rule-based KindFlag values (SKILL / USE_SKILL / PASSIVE) are
    excluded — those have deterministic producers outside the
    compactor and the LLM has no business emitting them.

    Trigger definitions (compactor prompt teaches these — keep
    aligned with the prompt template):

      - ``USER_FEEDBACK`` — the turn carries behaviour-shaping signal
        from the user: approval, correction, preference, rejection.
        Lives on user-origin nodes. Sentiment direction is NOT
        encoded here; consumers read the body or check the following
        assistant turn for ``SUCCESS``.

      - ``USER_REJECT`` — USER_FEEDBACK subset; the user explicitly
        rejected, corrected, or negated the prior assistant turn.
        Fast-path filter for "avoid past failure" queries. No
        symmetric USER_APPROVE — "what worked" lives on SUCCESS on
        the assistant turn.

      - ``SUCCESS`` — this assistant turn's action achieved its
        stated effect, verified by downstream signal in the same
        L1 buffer (no follow-up correction, USER_FEEDBACK positive
        sentiment, or explicit tool-side confirmation). Retroactive
        on the assistant turn that produced the effect.

      - ``COMPLETE`` — this turn closes out a discrete task / phase /
        project. Compactor applies on judgment-laden cases (user
        signals closure in their turn, or assistant verifies a unit
        of work is done). Rule-based COMPLETE attachment (archive,
        task done) lives outside the compactor.
    """
    USER_FEEDBACK = 'FLAG:USER_FEEDBACK'
    USER_REJECT = 'FLAG:USER_REJECT'
    SUCCESS = 'FLAG:SUCCESS'
    COMPLETE = 'FLAG:COMPLETE'
