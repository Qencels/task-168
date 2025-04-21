"""Base class for MWS agents."""

import logging
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, Any, Tuple


class BaseMwsAgent(ABC):
    """Abstract base class for all MWS agents."""

    def __init__(self, role: str, goal: str, backstory: str):
        """
        Initialize the base agent.

        Args:
            role (str): Role of the agent.
            goal (str): Goal of the agent.
            backstory (str): Backstory of the agent.
        """
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.lock = Lock()

    @abstractmethod
    def run(self, query: str, shared_context: Dict[str, Any]) -> Tuple[str, float]:
        """
        Run the agent with the given query and shared context.

        Args:
            query (str): The user query extracted from the context.
            shared_context (Dict[str, Any]): Shared context between agents.

        Returns:
            Tuple[str, float]: Agent result and execution time.
        """
        pass

    def _log_and_measure_time(self, query: str, execution_logic) -> Tuple[str, float]:
        """Helper method to log start/end and measure execution time."""
        start_time = time.time()

        actual_query = query
        if "Последний запрос клиента:" in query:
            query_start = query.find("Последний запрос клиента: ") + len("Последний запрос клиента: ")
            query_end = query.find("\"", query_start)
            if query_start != -1 and query_end != -1:
                actual_query = query[query_start:query_end].strip("\"")

        print(f"\n[Агент: {self.role}] Начало обработки запроса '{actual_query}'")
        logging.info(f"[Агент: {self.role}] Начало обработки запроса '{actual_query}'")

        result = execution_logic()

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"[Агент: {self.role}] Результат: {str(result)[:100]}...")
        print(f"[Агент: {self.role}] Время выполнения: {execution_time:.2f} секунд")
        logging.info(f"[Агент: {self.role}] Результат: {str(result)[:100]}...")
        logging.info(f"[Агент: {self.role}] Время выполнения: {execution_time:.2f} секунд")

        return result, execution_time
