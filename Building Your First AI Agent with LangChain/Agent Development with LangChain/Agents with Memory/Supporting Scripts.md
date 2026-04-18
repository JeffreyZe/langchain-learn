# Guide for Demo15

By default, agents are stateless,which means every call is treated like a completely new conversation. We will enable our agent to become smarter and prioritize action items better by adding memory to our create_agent framework, using *InMemorySaver*.

Integrating **InMemorySaver** will enable the:
- Messages to persist between agent calls
- Agent to “remember” earlier interactions in the same session
- Clean separation of conversations using thread_id

## New Imports
from langgraph.checkpoint.memory import InMemorySaver

## Rewrite the SYSTEM_PROMPT

***Previous Prompt:***

You are an Intelligent Meeting Notes Assistant.

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

***New Prompt:***

You are an Intelligent Meeting Notes Assistant.

This agent works across multiple messages in the same session, and must remember:
- What the user said earlier
- What you previously explained
- The meeting transcript you already analyzed

If the user asks a question like:
• “Can you remind me what the meeting was about?”
• “What did you extract earlier?”
• “What decisions did we identify?”

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

## In the main.py

1. We will create an instance of memory object using *memory = InMemorySaver()*
2. Then, we will attach this to the agemt using the *checkpointer* parameter
3. Finally, we will start a session using thread_id making sure that memory persists only inside that session.

## Final Invocation
This time when we invoke the agent, the following happens:
- First call: Analyze transcript
- Second call: User can ask any follow-up question. Here the agent will recall previous context. 

