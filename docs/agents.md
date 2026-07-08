# Agent Architecture & Debate Protocol

## Multi-Agent System (MAS) Concept
The system utilizes a multi-agent LLM architecture built on LangGraph to simulate short- and long-term investment decision-making. Specialized agents debate a stock's outlook across different financial dimensions, overseen by a Portfolio Manager and evaluated by a Critic[cite: 1]. System prompts are dynamically loaded via `prompt_loader.py` from YAML configuration files.

## Agent Roles

*   **Portfolio Manager (PM):** The central node of the graph. It asks questions to the specialized agents, gathers their reports, and synthesizes an overall opinion supported by arguments.
*   **Critic:** Evaluates the Portfolio Manager's synthesized opinion. It acts as a safeguard, ensuring the PM's logic is sound and unbiased.
*   **Financial Agent:** Analyzes the stock's financial fundamentals and provides reasoning.
*   **Sentiment Agent:** Evaluates market sentiment and provides an opinion.
*   **Macro Agent:** Analyzes the broader macroeconomic context and provides an opinion.

## The Debate Protocol (Execution Graph)

The decision-making process follows a strict iterative debate protocol governed by LangGraph:

1.  **Initiation:** The Portfolio Manager asks the 3 specialized agents (Financial, Sentiment, Macro) to analyze the stock.
2.  **Agent Reasoning:** The agents perform their reasoning and return their individual opinions and reports to the PM.
3.  **Synthesis:** The Portfolio Manager creates an initial opinion formulated with arguments and the reports gathered from the worker agents.
4.  **Evaluation:** The Critic checks the Portfolio Manager's opinion with agent reports.
5.  **Termination Condition:** 
    *   **If True:** If the Critic agrees with the PM, OR if the debate reaches the strict 3-round limit, the loop terminates and the final prediction is returned.
    *   **If False:** If the Critic disagrees and the round limit has not been reached, the Portfolio Manager uses the feedback from the Critic to ask the 3 agents again. *(Note: The PM may decide to only query specific agents in subsequent rounds based on the Critic's feedback).*