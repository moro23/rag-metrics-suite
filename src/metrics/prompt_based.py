import logging
import re

from src.metrics.base import RAGMetric, SupervisedMetric

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, answer_similarity, answer_correctness
    from tonic_validate.metrics import AnswerConsistencyMetric, AnswerSimilarityMetric as TonicSimilarityMetric
    from tonic_validate import ValidateScorer, LLMResponse, BenchmarkItem
    from transformers import pipeline
    _DEPS_AVAILABLE = True
except ImportError:
    logging.warning("Ragas/Tonic/transformers not found - some prompt-based metrics unavailable")
    _DEPS_AVAILABLE = False


# -- Ragas wrappers (stubs - not used in the low-memory demo) --

class RagasFaithfulness(RAGMetric):
    __name__ = "RagasFaithfulness"
    def __init__(self, ragas_llm): self.metric = faithfulness; self.metric.llm = ragas_llm
    def evaluate(self, question: str, context: str, answer: str, **kwargs) -> float: return 0.0

class RagasRelevancy(RAGMetric):
    __name__ = "Relevancy"
    def __init__(self, ragas_llm): self.metric = answer_relevancy; self.metric.llm = ragas_llm
    def evaluate(self, question: str, context: str, answer: str, **kwargs) -> float: return 0.0

class RagasSimilarity(SupervisedMetric):
    __name__ = "RSim"
    def __init__(self, ragas_llm): self.metric = answer_similarity; self.metric.llm = ragas_llm
    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float: return 0.0

class RagasAnswerCorrectness(SupervisedMetric):
    __name__ = "RagasAnswerCorrectness"
    def __init__(self, ragas_llm): self.metric = answer_correctness; self.metric.llm = ragas_llm
    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float: return 0.0


# -- Tonic wrappers (also stubs for now) --

class TonicConsistency(RAGMetric):
    __name__ = "Consistency"
    def __init__(self): self.scorer = ValidateScorer([AnswerConsistencyMetric()])
    def evaluate(self, question: str, context: str, answer: str, **kwargs) -> float: return 0.0

class TonicSimilarity(SupervisedMetric):
    __name__ = "TSim"
    def __init__(self): self.scorer = ValidateScorer([TonicSimilarityMetric()])
    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float: return 0.0


# -- Custom LLM judge using a local model --

class CustomLLMJudge(SupervisedMetric):
    __name__ = "CustomLLMJudge"

    def __init__(self, model_name: str = "google/flan-t5-small"):
        if not _DEPS_AVAILABLE:
            raise ImportError("transformers is required for CustomLLMJudge")

        logging.info(f"Loading judge model: {model_name}")
        self.pipe = pipeline("text2text-generation", model=model_name)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0

        prompt = (
            f'Score the generated answer on a scale from 1 to 5. '
            f'Reference Answer: "{correct_answer}" '
            f'Generated Answer: "{answer}" Score:'
        )
        output = self.pipe(prompt, max_length=10)[0]['generated_text']

        try:
            match = re.search(r'\d+\.?\d*', output)
            if not match:
                return 0.0
            # normalize 1-5 range to 0-1
            return (float(match.group()) - 1) / 4
        except Exception:
            return 0.0