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
        description='List of questions derived from original content to evaluate RAG results.'
    )


class Memory(TypedDictSchema):
    forget: Set[str] = Field(
        default=set(),
        description='List of existing ids of memories to overwrite or forget.'
    )
    save: List[Dict] = Field(
        default = ...,
        description='List of new memories to save.'
    )


class SummaryAgentStateSchema(TypedDictSchema):
    action: Memory = Field(
        default = ...,
        description='Memory action to optimize RAG.'
    )


class ValidateAgentStateSchema(TypedDictSchema):
    answers: List[Optional[str]] = Field(
        default = ...,
        description=(
            'List of answers to queries derived from RAG results. '
            'If some answers are None, it means that they cannot be determined from current RAG results.'
        )
    )


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
