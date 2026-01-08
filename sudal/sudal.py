from ato.adict import ADict
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from sudal.schemas import (
    SUDALStateSchema,
    QueryAgentStateSchema,
    SummaryAgentStateSchema,
    ValidateAgentStateSchema,
    DecisionAgentStateSchema,
    QueryAgentResponseSchema,
    SummaryAgentResponseSchema,
    ValidateAgentResponseSchema,
    DecisionAgentResponseSchema
)


class SUDAL:
    def __init__(self, config):
        self._config = config
        self.graph = None
        self.agents = ADict()
        self.schemas = ADict()
        self._build_graph()

    # inherit SUDAL and change schemas for specific use
    def _init_schemas(self):
        self.schemas.update(
            query_agent=ADict(
                state_schema=QueryAgentStateSchema,
                response_schema=QueryAgentResponseSchema
            ),
            summary_agent=ADict(
                state_schema=SummaryAgentStateSchema,
                response_schema=SummaryAgentResponseSchema
            ),
            validate_agent=ADict(
                state_schema=ValidateAgentStateSchema,
                response_schema=ValidateAgentResponseSchema
            ),
            decision_agent=ADict(
                state_schema=DecisionAgentStateSchema,
                response_schema=DecisionAgentResponseSchema
            )
        )

    def _build_graph(self):
        graph = StateGraph(SUDALStateSchema)
        self._init_schemas()

        self.agents.query_agent = create_agent(
            model=self.config.agents.query_agent,
            state_schema=self.schemas.query_agent.state_schema,
            response_format=ToolStrategy(self.schemas.query_agent.response_schema)
        )
        self.agents.summary_agent = create_agent(
            model=self.config.agents.summary_agent,
            state_schema=self.schemas.summary_agent.state_schema,
            response_format=ToolStrategy(self.schemas.summary_agent.response_schema)
        )
        self.agents.validate_agent = create_agent(
            model=self.config.agents.validate_agent,
            state_schema=self.schemas.validate_agent.state_schema,
            response_format=ToolStrategy(self.schemas.validate_agent.response_schema)
        )
        self.agents.decision_agent = create_agent(
            self.config.agents.decision_agent,
            state_schema=self.schemas.decision_agent.state_schema,
            response_format=ToolStrategy(self.schemas.decision_agent.response_schema)
        )

        graph.add_node('summarize', self.summarize)
        graph.add_node('forget_memory', self.forget_memory)
        graph.add_node('save_memory', self.save_memory)
        graph.add_node('generate_queries', self.generate_queries)
        graph.add_node('search', self.search)
        graph.add_node('validate', self.validate)
        graph.add_node('make_decision', self.make_decision)

        graph.add_edge(START, 'summarize')
        graph.add_edge('summarize', 'forget_memory')
        graph.add_edge('forget_memory', 'save_memory')
        graph.add_edge('save_memory', 'generate_queries')
        graph.add_edge('generate_queries', 'search')
        graph.add_edge('search', 'validate')
        graph.add_edge('validate', 'make_decision')
        graph.add_conditional_edges(
            'make_decision',
            self.route,
            dict(delegate='summarize', finish=END)
        )

        self.graph = graph.compile()

    @property
    def config(self):
        return self._config

    def summarize(self, state):
        # summarize given contents and determine which memory is forgotten and memorized
        pass

    def forget_memory(self, state):
        # remove memory from db
        pass

    def save_memory(self, state):
        # add memory to db
        pass

    def generate_queries(self, state):
        # generate adversarial queries
        pass

    def search(self, state):
        # search via RAG
        pass

    def validate(self, state):
        # answer queries using searched results only to validate
        pass

    def make_decision(self, state):
        # make decision via validation results
        pass

    def route(self, state):
        pass

    def invoke(self, task):
        pass
