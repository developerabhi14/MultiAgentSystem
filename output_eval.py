import os
import logging

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile")

LOG_DIR="logs"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "execution.log"),
    level=logging.INFO,
    format="%(asctime)s- %(levelname)s - %(message)s",
    filemode="w"
)

logger=logging.getLogger(__name__)

def write_text_file(filename:str, content: str):
    filepath=os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def planner_agent(state):
    logger.info("Planner agent started")
    logger.info(f"User query: {state['user_query']}")
    prompt=f"""
    You are a planning agent ina multi agent AI system.

    Your job is to read the user's query and create a short plan for how the worker agent should answer.

    User query: {state['user_query']}

    Return only a short actionable plan
    """
    response=llm.invoke(prompt)
    plan=response.content if hasattr(response, "content") else str(response)
    state['plan']=plan
    write_text_file("planner_output.txt", plan)
    logger.info(f"Planner output:{plan}")
    return state


def worker_agent(state):
    state['worker_calls']+=1
    feedback=state.get('review_reason',"")
    logger.info(f"Woker call #{state['worker_calls']}")
    prompt=f"""
    You are a worker agent

    User query:{state['user_query']}

    Plan: {state['plan']}

    Previous reviewer feedback: {feedback}

    Write the best possible improved respinse. If reviewer feedback exists,explicitly fix those issues

    """
    response=llm.invoke(prompt)

    draft_response=response.content if hasattr(response, "content") else str(response)

    state['draft_response']=draft_response

    write_text_file(f"Worker_output_{state['worker_calls']}.txt", draft_response)

    logger.info(f"Worker_output_{state['worker_calls']}: {draft_response}")

    return state

def reviewer_agent(state):
    state['reviewer_calls']+=1
    prompt=f"""
    You are a Strict reviewer agent.

    User query: {state['user_query']},

    Draft response: {state['draft_response']}

    check for :
    - concrete examples
    - implementation details
    - tradeoffs
    - clarity
    - actionable recommendations

    If anything is missing, revise.
    Return EXACTLY in this format:
    Decision: approve OR revise
    Reason: brief reason
    """
    response=llm.invoke(prompt)
    raw_output=response.content.strip() if hasattr(response, "content") else str(response)
    
    decision= "approve" if "decision: approve" in raw_output.lower() else 'revise'

    reason_line=next((line for line in raw_output.splitlines() if line.lower().startswith("reason:")),"")
    reason=reason_line.replace("Reason","").strip() if reason_line else "No reason provided"
    state['review_decision']=decision
    state['review_reason']=reason

    write_text_file(
    f"reviewer_output_{state['reviewer_calls']}.txt",
    raw_output + f"\nParsed Decision: {decision}\nReason: {reason}"
    )
    logger.info(f"Reviewer decision #{state['reviewer_calls']}: {decision} ({reason})")
    return state

def review_router(state):
    if state.get("review_decison")=="approve" or state.get("review_count",0)>=2:
        return "__end__"
    state['revision_count']=state.get("revision_count", 0)+1
    return "worker_agent"


workflow=StateGraph(dict)

workflow.add_node("planner_agent", planner_agent)
workflow.add_node("worker_agent", worker_agent)
workflow.add_node("reviewer_agent", reviewer_agent)

workflow.set_entry_point("planner_agent")
workflow.add_edge('planner_agent', "worker_agent")
workflow.add_edge("worker_agent", "reviewer_agent")
workflow.add_conditional_edges(
    "reviewer_agent",review_router,
    {"worker_agent": "worker_agent", 
     "__end__":END}
)

app=workflow.compile()

try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    logger.info("Graph PNG saved as graph.png")
except Exception as e:
    logger.error(f"Failed to generate graph PNG: {e}")

user_query=input("Enter your query")
logger.info(f"User input: {user_query}")
initial_state = {
    "user_query": user_query,
    "plan": "",
    "draft_response": "",
    "review_reason": "",
    "review_decision": "",
    "worker_calls": 0,
    "reviewer_calls": 0,
    "revision_count": 0
}
result=app.invoke(initial_state)
final_output=result.get("draft_response","")
write_text_file("final_output.txt", final_output)

print("\n=== FINAL RESPONSE ===")
print(final_output)

logger.info(f"Final output: {final_output}")