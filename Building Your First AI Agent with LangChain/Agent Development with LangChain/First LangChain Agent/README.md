# Demonstration 13: Building Your First LangChain Agent

### Scenario

Imagine you are a project manager in a fast-moving organization where frequent meetings generate valuable information, yet most of it remains buried in long transcripts or scattered notes. Key decisions, follow-ups, and responsibilities are often missed. 

To streamline this process, you build an Intelligent Meeting Notes Agent, an assistant that automatically analyzes transcripts, highlights key discussion points, extracts decisions, identifies action items, and produces a clean, structured summary for your team.

### Problem Statement

Given a meeting transcript, create an autonomous agent that can extract decisions, identify action items, and produce a concise, structured summary—while optionally leveraging memory to maintain continuity across meetings.

### What You Will Learn

* How to construct an autonomous agent using create\_agent in LangChain.
* How to design a role-defining system prompt using ChatPromptTemplate.
* How to integrate a Gemini model as the agent’s reasoning engine.
* How to define and use custom tools using the python functions.

### System Prompt

You are an Intelligent Meeting Notes Assistant.



Your role is to analyze meeting transcripts and generate:

1\. Key discussion points

2\. Decisions taken

3\. Action items with owners

4\. A clear, structured summary



Use the available tools when necessary to fetch transcripts, extract actions,

identify decisions, and produce summaries. Maintain accuracy and avoid

inventing information.



If memory is available, incorporate relevant details from previous meetings,

such as participants, project context, prior decisions, and ongoing follow-ups.

### Tools

1. extract\_key\_points(text)
2. extract\_action\_items(text)
3. summarize\_meeting(text) 



