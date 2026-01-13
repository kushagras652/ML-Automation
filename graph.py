from langgraph.graph import StateGraph,END
from state import AgentState

from agents.orchestration import orchestrator_agent
from agents.eda_agent import eda_agent
from agents.ml_agent import ml_agent
from agents.insight_agent import insight_agent

def build_graph():
    graph=StateGraph(AgentState)

    graph.add_node('orchestrator',orchestrator_agent)
    graph.add_node('eda',eda_agent)
    graph.add_node('ml',ml_agent)
    graph.add_node('insight',insight_agent)

    graph.set_entry_point('orchestrator')
    graph.add_edge('orchestrator','eda')
    graph.add_edge('eda','ml')
    graph.add_edge('ml','insight')
    graph.add_edge('insight',END)


    return graph.compile()