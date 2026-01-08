from typing import Set, List, Dict, Optional

from pydantic import Field
from pydantic_core.core_schema import TypedDictSchema


class FailureCategories(TypedDictSchema):
    pass


class SUDALStateSchema(TypedDictSchema):
    pass


class QueryAgentStateSchema(TypedDictSchema):
    questions: Optional[List[str]] = Field(
        default=None,
        description='List of questions derived from original content to validate RAG results.'
    )


class Memory(TypedDictSchema):
    forget: Set[str]
    save: List[Dict]


class SummaryAgentStateSchema(TypedDictSchema):
    action: Memory


class ValidateAgentStateSchema(TypedDictSchema):
    answers: List[str]


class DecisionAgentStateSchema(TypedDictSchema):
    pass


class QueryAgentResponseSchema(TypedDictSchema):
    pass


class SummaryAgentResponseSchema(TypedDictSchema):
    pass


class ValidateAgentResponseSchema(TypedDictSchema):
    pass


class DecisionAgentResponseSchema(TypedDictSchema):
    pass
