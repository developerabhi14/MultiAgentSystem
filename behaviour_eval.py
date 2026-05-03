import os
import logging

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# ---------------- LOGGING ---------------- #
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "execution.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

logger = logging.getLogger(__name__)

def write_text_file(filename: str, content: str):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

# ---------------- TOOLS ---------------- #
def search_tool(query):
    return f"[SEARCH RESULT] Info about: {query}"

def code_tool(task):
    return f"[CODE EXECUTED] Solved: {task}"

def db_tool(query):
    return f"[DB RESULT] Retrieved structured data for: {query}"

# ---------------- PLANNER ---------------- #
def planner_agent(state):
    prompt = f"""
    You are a planning agent.

    Create a short actionable plan.

    User query: {state['user_query']}
    """
    response = llm.invoke(prompt)
    state['plan'] = response.content if hasattr(response, "content") else str(response)

    write_text_file("planner_output.txt", state['plan'])
    return state

# ---------------- WORKER (BEHAVIOR FOCUSED) ---------------- #
def worker_agent(state):
    state['worker_calls'] += 1

    feedback = (
        state.get('behavior_feedback', "") + "\n" +
        state.get('reasoning_feedback', "") + "\n" +
        state.get('review_reason', "")
    )

    action_trace = []

    prompt = f"""
    You are an AI agent with tools:

    - search_tool → use for factual information
    - code_tool → use for calculations or transformations
    - db_tool → use for structured queries

    STRICT RULES:
    - Choose correct tool
    - Do NOT hallucinate if tool needed
    - One tool per step

    Format:

    Thought:
    Action:
    Action Input:

    Repeat if needed.

    Then:

    FINAL ANSWER:
    <answer>

    REASONING TRACE:
    - Step 1:
    - Step 2:

    User query: {state['user_query']}
    Plan: {state['plan']}
    Feedback: {feedback}
    """

    response = llm.invoke(prompt)
    output = response.content if hasattr(response, "content") else str(response)

    lines = output.splitlines()

    final_answer = ""
    reasoning_trace = ""
    current_action = None

    for i, line in enumerate(lines):
        line_lower = line.lower()

        if line_lower.startswith("action:"):
            current_action = line.split(":", 1)[1].strip()

        elif line_lower.startswith("action input:"):
            current_input = line.split(":", 1)[1].strip()

            if current_action:
                if "search_tool" in current_action:
                    result = search_tool(current_input)
                elif "code_tool" in current_action:
                    result = code_tool(current_input)
                elif "db_tool" in current_action:
                    result = db_tool(current_input)
                else:
                    result = "Unknown tool"

                action_trace.append({
                    "action": current_action,
                    "input": current_input,
                    "result": result
                })

        elif "final answer:" in line_lower:
            final_answer = line.split(":", 1)[1].strip()

        elif "reasoning trace:" in line_lower:
            reasoning_trace = "\n".join(lines[i+1:]).strip()

    state['draft_response'] = final_answer
    state['reasoning_trace'] = reasoning_trace
    state['action_trace'] = action_trace

    write_text_file(f"worker_output_{state['worker_calls']}.txt", output)
    write_text_file(f"action_trace_{state['worker_calls']}.txt", str(action_trace))

    return state

# ---------------- BEHAVIOR EVALUATOR ---------------- #
def behavior_evaluator_agent(state):
    prompt = f"""
    Evaluate agent behavior.

    Focus on:
    - Correct tool selection
    - Missing tool usage
    - Unnecessary actions
    - Efficiency
    - Logical order

    Tools:
    - search_tool → factual
    - code_tool → computation
    - db_tool → structured

    User Query:
    {state['user_query']}

    Action Trace:
    {state.get('action_trace', [])}

    Rules:
    - Wrong tool → major penalty
    - No tool when needed → penalty
    - Extra steps → penalty

    Return EXACTLY:

    Score: <0-10>
    Decision: approve OR revise
    Reason: <short explanation>
    """

    response = llm.invoke(prompt)
    raw = response.content.strip() if hasattr(response, "content") else str(response)

    decision = "approve" if "approve" in raw.lower() else "revise"

    reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No reason"

    state['behavior_decision'] = decision
    state['behavior_feedback'] = reason

    write_text_file("behavior_eval.txt", raw)

    return state

# ---------------- REASONING EVALUATOR ---------------- #
def reasoning_evaluator_agent(state):
    prompt = f"""
    Evaluate reasoning.

    Query:
    {state['user_query']}

    Reasoning:
    {state.get('reasoning_trace', '')}

    Return EXACTLY:

    Score: <0-10>
    Decision: approve OR revise
    Reason: <short explanation>
    """

    response = llm.invoke(prompt)
    raw = response.content.strip() if hasattr(response, "content") else str(response)

    decision = "approve" if "approve" in raw.lower() else "revise"

    reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No reason"

    state['reasoning_decision'] = decision
    state['reasoning_feedback'] = reason

    write_text_file("reasoning_eval.txt", raw)

    return state

# ---------------- OUTPUT REVIEWER ---------------- #
def reviewer_agent(state):
    state['reviewer_calls'] += 1

    prompt = f"""
    Evaluate final answer quality.

    Query:
    {state['user_query']}

    Answer:
    {state['draft_response']}

    Return EXACTLY:

    Decision: approve OR revise
    Reason: <short explanation>
    """

    response = llm.invoke(prompt)
    raw = response.content.strip() if hasattr(response, "content") else str(response)

    decision = "approve" if "approve" in raw.lower() else "revise"

    reason_line = next((l for l in raw.splitlines() if l.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason:", "").strip() if reason_line else "No reason"

    state['review_decision'] = decision
    state['review_reason'] = reason

    write_text_file(f"review_{state['reviewer_calls']}.txt", raw)

    return state

# ---------------- ROUTER ---------------- #
def review_router(state):
    if (
        state.get("review_decision") == "approve" and
        state.get("reasoning_decision") == "approve" and
        state.get("behavior_decision") == "approve"
    ) or state.get("revision_count", 0) >= 2:
        return "__end__"

    state['revision_count'] += 1
    return "worker"

# ---------------- GRAPH ---------------- #
workflow = StateGraph(dict)

workflow.add_node("planner", planner_agent)
workflow.add_node("worker", worker_agent)
workflow.add_node("behavior_eval", behavior_evaluator_agent)
workflow.add_node("reasoning_eval", reasoning_evaluator_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "worker")
workflow.add_edge("worker", "behavior_eval")
workflow.add_edge("behavior_eval", "reasoning_eval")
workflow.add_edge("reasoning_eval", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    review_router,
    {
        "worker": "worker",
        "__end__": END
    }
)

app = workflow.compile()

# ---------------- RUN ---------------- #
user_query = input("Enter query: ")

initial_state = {
    "user_query": user_query,
    "plan": "",
    "draft_response": "",
    "reasoning_trace": "",
    "action_trace": [],
    "review_reason": "",
    "reasoning_feedback": "",
    "behavior_feedback": "",
    "review_decision": "",
    "reasoning_decision": "",
    "behavior_decision": "",
    "worker_calls": 0,
    "reviewer_calls": 0,
    "revision_count": 0
}

result = app.invoke(initial_state)

print("\n=== FINAL OUTPUT ===")
print(result.get("draft_response", ""))
