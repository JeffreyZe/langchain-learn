# Demonstration 14: Enhancing Agents with Memory

### Scenario

After building the base Intelligent Meeting Notes Agent, we now want to enable the agent to maintain continuity across recurring meetings. In many organizations, discussions span multiple sessions in which projects evolve, follow-ups recur, and decisions depend on previous context. Without memory, the agent treats every meeting as an isolated event, losing valuable historical information.

#### Problem Statement

Extend the existing Intelligent Meeting Notes Agent by adding a memory component that can store and retrieve past summaries, action items, participant roles, and ongoing project details. The memory-enhanced agent should be able to reference prior meetings to generate richer, more context-aware outputs.

#### What You Will Learn

* Introduce memory as an enhancement layer on top of the existing create\_agent framework.
* Implement simple memory storage (e.g., InMemoryStore)
* Modify the agent prompt to incorporate retrieved memory
* Run multiple meeting invocations to demonstrate accumulated context
* Compare outputs with vs. without memory to show the improvement.



Method: Simple short-term memory in create\_agent using InMemorySaver: “Enable memory” without touching advanced graph concepts



