from langgraph.graph import StateGraph
from src.agents.nodes import researcher_node, bull_node, bear_node, pm_node
from src.agents.state import AgentState

graph = StateGraph(AgentState)

graph.add_node("researcher", researcher_node)
graph.add_node("bull", bull_node)
graph.add_node("bear", bear_node)
graph.add_node("pm", pm_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "bull")
graph.add_edge("researcher", "bear")
graph.add_edge("bull", "pm")
graph.add_edge("bear", "pm")

app_graph = graph.compile()