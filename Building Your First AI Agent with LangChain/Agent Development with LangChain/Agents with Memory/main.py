import os
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from encoding import read_transcript

# New Import for Memory
from langgraph.checkpoint.memory import InMemorySaver 

load_dotenv()

SYSTEM_PROMPT = """
You are an Intelligent Meeting Notes Assistant.

This agent works across multiple messages in the same session, and must remember:
- What the user said earlier
- What you previously explained
- The meeting transcript you already analyzed

If the user asks a question like:
- “Can you remind me what the meeting was about?”
- “What did you extract earlier?”
- “What decisions did we identify?”

– you should use your memory of previous steps to answer.

Your tasks:
- Extract key discussion points
- Extract decisions
- Extract action items and owners
- Create summaries when needed
- Support follow-up questions that depend on earlier context

Rules:
- Do not invent details not present in the transcript or prior messages
- Keep answers clear and structured
- Use tools when helpful
"""

def extract_key_points(text: str):
    """Extract lines containing discussions or updates."""
    
    lines = text.split("\n")
    points = [line.strip() for line in lines if "discuss" in line.lower() or "update" in line.lower()]
    return "\n".join(points) if points else "No key points found."


def extract_action_items(text: str):
    """Extract lines that mention action items."""
    lines = text.split("\n")
    actions = [line.strip() for line in lines if any(word in line.lower() for word in ["action", "will", "needs", "should"])]
    return "\n".join(actions) if actions else "No action items found."


def summarize_meeting(text: str):
    """Return a simple short summary of the meeting."""
    lines = text.split("\n")[:10]
    return f"Summary:\n" + "\n".join(lines)

# create an instace of memory
memory = InMemorySaver() 

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tools = [summarize_meeting, extract_action_items, extract_key_points]
agent = create_agent(
    model=model,   
    tools=tools,   
    system_prompt=SYSTEM_PROMPT,
    # Add short-term memory to the create_agent framework
    checkpointer=memory 
)

THREAD_ID = "meeting-session-1"

def main():
    transcript = read_transcript("transcript.txt")
    
    print("First, the agent will analyze the meeting transcript...\n")

    # First call — seed the memory with transcript
    initial = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Here is the meeting transcript. Please analyze it:\n\n"
                    + transcript
                )
            ]
        },
        {"configurable": {"thread_id": THREAD_ID}},
    )

    print("Agent:\n", initial["messages"][-1].content)
    print("\nNow you can ask follow-up questions.")
    print("Type 'exit' to stop.\n")

    # Interactive loop
    while True:
        user_input = input("You: ")

        if user_input.lower().strip() == "exit":
            print("\nEnding demo. Goodbye!\n")
            break

        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            {"configurable": {"thread_id": THREAD_ID}},
        )

        print("\nAgent:", response["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()

