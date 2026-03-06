import logging
import statistics
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

from scipy.stats import spearmanr

from src.metrics.base import Metric

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RAGExample:
    question: str
    context: str
    answer: str
    correct_answer: Optional[str] = None


@dataclass
class GenerationEvaluationResult:
    scores_raw: Dict[str, List[float]] = field(default_factory=dict)
    scores_mean: Dict[str, float] = field(default_factory=dict)
    scores_std: Dict[str, float] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)

    def to_string(self) -> str:
        report = ["=" * 50, " RAG Evaluation Report ", "=" * 50]

        report.append("\nAverage Metric Scores:")
        if self.scores_mean:
            sorted_means = sorted(self.scores_mean.items(), key=lambda x: x[1], reverse=True)
            for name, mean in sorted_means:
                std = self.scores_std.get(name, 0.0)
                report.append(f"  {name:<25}: {mean:.4f} (±{std:.4f})")
        else:
            report.append("  (none)")

        report.append("\nMetric Correlations (Spearman):")
        if self.correlations:
            sorted_corrs = sorted(self.correlations.items(), key=lambda x: x[1], reverse=True)
            for (m1, m2), rho in sorted_corrs:
                report.append(f"  {m1} / {m2:<30}: {rho:.4f}")
        else:
            report.append("  (none)")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


class EvaluationPipeline:
    def __init__(self, metrics: List[Metric]):
        if not metrics:
            raise ValueError("Need at least one metric.")
        self.metrics = metrics
        logger.info(f"Pipeline initialized with: {[m.__name__ for m in self.metrics]}")

    def run(self, dataset: List[RAGExample]) -> GenerationEvaluationResult:
        logger.info(f"Evaluating {len(dataset)} examples...")

        # score each example with each metric
        scores_raw = {m.__name__: [] for m in self.metrics}

        for metric in self.metrics:
            logger.info(f"  running {metric.__name__}")
            for i, ex in enumerate(dataset):
                try:
                    score = metric.evaluate(
                        question=ex.question,
                        context=ex.context,
                        answer=ex.answer,
                        correct_answer=ex.correct_answer,
                    )
                    scores_raw[metric.__name__].append(score)
                except Exception as e:
                    logger.warning(f"  {metric.__name__} failed on example {i}: {e}")
                    scores_raw[metric.__name__].append(None)

        # aggregate stats
        scores_mean = {}
        scores_std = {}
        for name, scores in scores_raw.items():
            valid = [s for s in scores if isinstance(s, (int, float))]
            if not valid:
                logger.warning(f"  no valid scores for {name}")
                scores_mean[name] = 0.0
                scores_std[name] = 0.0
                continue
            scores_mean[name] = statistics.mean(valid)
            scores_std[name] = statistics.stdev(valid) if len(valid) > 1 else 0.0

        # pairwise spearman correlations
        correlations = {}
        metric_names = list(scores_raw.keys())
        for m1, m2 in combinations(metric_names, 2):
            # only compare where both metrics produced a score
            paired = [
                (s1, s2) for s1, s2 in zip(scores_raw[m1], scores_raw[m2])
                if s1 is not None and s2 is not None
            ]
            if len(paired) < 2:
                continue
            a, b = zip(*paired)
            rho, _ = spearmanr(a, b)
            # guard against NaN
            correlations[(m1, m2)] = rho if rho == rho else 0.0

        logger.info("Evaluation complete.")
        return GenerationEvaluationResult(
            scores_raw=scores_raw,
            scores_mean=scores_mean,
            scores_std=scores_std,
            correlations=correlations,
        )