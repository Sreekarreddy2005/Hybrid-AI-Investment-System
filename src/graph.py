# graph.py
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import researcher_node, bull_node, bear_node, pm_node

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("bull", bull_node)
workflow.add_node("bear", bear_node)
workflow.add_node("pm", pm_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "bull")
workflow.add_edge("researcher", "bear")
workflow.add_edge("bull", "pm")
workflow.add_edge("bear", "pm")
workflow.add_edge("pm", END)

app_graph = workflow.compile()
