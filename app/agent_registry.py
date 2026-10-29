"""Agent registration and task dispatch for SALAZAR."""

from app.agents import Agent, TaskResult
from app.task_router import RoutingDecision, TaskType


class AgentRegistry:
    """Register agents and dispatch tasks to the appropriate agent."""

    def __init__(self):
        self._agents: dict[TaskType, Agent] = {}

    def register(self, task_type: TaskType, agent: Agent) -> None:
        if not isinstance(task_type, TaskType):
            raise TypeError("task_type must be a TaskType.")

        if not isinstance(agent, Agent):
            raise TypeError("agent must be an Agent instance.")

        if task_type.value != agent.task_type:
            raise ValueError(
                f"Agent task_type '{agent.task_type}' does not match "
                f"registration type '{task_type.value}'."
            )

        if task_type in self._agents:
            raise ValueError(
                f"An agent is already registered for '{task_type.value}'."
            )

        self._agents[task_type] = agent

    def get(self, task_type: TaskType) -> Agent:
        if not isinstance(task_type, TaskType):
            raise TypeError("task_type must be a TaskType.")

        try:
            return self._agents[task_type]
        except KeyError:
            raise LookupError(
                f"No agent registered for '{task_type.value}'."
            ) from None

    def dispatch(self, decision: RoutingDecision, task: str) -> TaskResult:
        if not isinstance(decision, RoutingDecision):
            raise TypeError("decision must be a RoutingDecision.")

        agent = self.get(decision.task_type)
        result = agent.execute(task)

        if not isinstance(result, TaskResult):
            raise TypeError(
                f"Agent '{agent.name}' did not return a TaskResult."
            )

        if result.agent_name != agent.name:
            raise ValueError(
                f"Result agent_name '{result.agent_name}' does not match "
                f"registered agent '{agent.name}'."
            )

        if result.task_type != decision.task_type.value:
            raise ValueError(
                f"Result task_type '{result.task_type}' does not match "
                f"routed task type '{decision.task_type.value}'."
            )

        return result
