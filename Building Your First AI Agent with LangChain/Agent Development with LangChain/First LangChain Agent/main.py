import os
from unittest import result
from dotenv import load_dotenv  ## Used to Load API key
from langchain_google_genai import ChatGoogleGenerativeAI  ## Used in Step 3
from langchain.agents import create_agent  ## Used in Step 4
from langchain_core.messages import HumanMessage
from encoding import read_transcript

# Load environment variables from .env file
load_dotenv()

## Step 1: Defining Agent Behaviour
SYSTEM_PROMPT = """
You are an Intelligent Meeting Notes Assistant.

Your responsibilities:
- Identify key discussion points from the meeting transcript
- Extract decisions made during the meeting
- Extract action items along with responsible owners
- Produce a clean, structured summary

Rules:
- Use tools when necessary
- Do NOT add information that is not in the transcript
- Be accurate, factual, and concise
"""


# Step 2: Defining Tools that LLM can use
def extract_key_points(text: str):
    """Extract lines containing discussions or updates."""

    lines = text.split("\n")
    points = [
        line.strip()
        for line in lines
        if "discuss" in line.lower() or "update" in line.lower()
    ]
    return "\n".join(points) if points else "No key points found."


def extract_action_items(text: str):
    """Extract lines that mention action items."""
    lines = text.split("\n")
    actions = [
        line.strip()
        for line in lines
        if any(word in line.lower() for word in ["action", "will", "needs", "should"])
    ]
    return "\n".join(actions) if actions else "No action items found."


def summarize_meeting(text: str):
    """Return a simple short summary of the meeting."""
    lines = text.split("\n")[:10]
    return f"Summary:\n" + "\n".join(lines)


## Step 3: Integrating an LLM model to power the Agent
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


## Step 4: Use the create_agent Framework

# Create a tool list to let LLM know what tools are at their disposal
tools = [summarize_meeting, extract_action_items, extract_key_points]

# Finally create the agent
agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


def main():

    # Read the transcript from a file
    transcript = read_transcript("transcript.txt")

    print("\n--- Agent Output ---\n")

    # invoke
    result = agent.invoke({"messages": [HumanMessage(content=transcript)]})

    # Print the raw output to understand the structure
    for msg in result["messages"]:
        print(type(msg), msg)

    content = result["messages"][-1].content
    if isinstance(content, str):
        print(content)
    elif isinstance(content, list):
        seen = set()
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text = block["text"].strip()
                if text and text not in seen:
                    seen.add(text)
                    parts.append(text)
        print("\n".join(parts))


if __name__ == "__main__":
    main()
