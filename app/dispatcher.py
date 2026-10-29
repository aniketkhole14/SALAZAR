"""Route SALAZAR requests to registered specialized agents."""

from app.agent_registry import AgentRegistry
from app.agents import TaskResult
from app.task_router import TaskRouter, RoutingDecision


class TaskDispatcher:
    """Classify a request and dispatch it to its registered agent."""

    def __init__(
        self,
        router: TaskRouter,
        registry: AgentRegistry,
    ):
        if not isinstance(router, TaskRouter):
            raise TypeError("router must be a TaskRouter.")
        if not isinstance(registry, AgentRegistry):
            raise TypeError("registry must be an AgentRegistry.")

        self.router = router
        self.registry = registry

    def dispatch(self, query: str) -> tuple[RoutingDecision, TaskResult]:
        """Route and execute a request through the agent registry."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty.")

        decision = self.router.route(query)
        result = self.registry.dispatch(decision, query)
        return decision, result
