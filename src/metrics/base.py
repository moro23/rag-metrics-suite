from abc import ABC, abstractmethod


class Metric(ABC):
    """Base class for all evaluation metrics."""
    __name__: str = "Metric"

    @abstractmethod
    def evaluate(self, **kwargs):
        pass


class SupervisedMetric(Metric):
    """Metric that compares a generated answer against a reference/correct answer."""

    @abstractmethod
    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        pass


class RAGMetric(Metric):
    """Metric that also takes the question and retrieved context as input."""

    @abstractmethod
    def evaluate(self, question: str, context: str, answer: str, **kwargs) -> float:
        pass