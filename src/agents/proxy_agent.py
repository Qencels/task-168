"""
Proxy agent to integrate MwsAgent with CrewAI framework
"""

from typing import Dict, Any

from agents.base_agent import BaseMwsAgent
from crewai import Agent


class ProxyAgent(Agent):
    """Proxy Agent to adapt MwsAgent for use with CrewAI framework"""

    class Config:
        extra = "allow"

    def __init__(self, mws_agent: BaseMwsAgent, shared_context: Dict[str, Any]):
        """
        Initialize ProxyAgent

        Args:
            mws_agent (BaseMwsAgent): An instance of a class derived from BaseMwsAgent.
            shared_context (Dict[str, Any]): Shared context dictionary passed between agents.
        """
        super().__init__(
            role=mws_agent.role,
            goal=mws_agent.goal,
            backstory=mws_agent.backstory,
            allow_delegation=False
        )
        self.mws_agent = mws_agent
        self.shared_context = shared_context

    def execute_task(self, task, context: str | None = None, tools: list | None = None) -> str:
        """
        Execute task using the wrapped MwsAgent's run method.

        Args:
            task: The task object assigned by CrewAI. Contains task description and other metadata.
            context (str | None): Additional context provided by CrewAI (e.g., from previous tasks).
            tools (list | None): Tools available to the agent (not used by MwsAgent directly).

        Returns:
            str: The result from the MwsAgent's run method.
        """

        task_description = task.description if hasattr(task, 'description') else str(task)

        agent_input = task_description
        if context:
            agent_input = (f"Context from previous steps: {context}\n"
                           f"---\n"
                           f"Task: {task_description}")

        result, _ = self.mws_agent.run(agent_input, self.shared_context)

        return str(result)
