import os
import nltk
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER
from nltk.translate.meteor_score import meteor_score

from src.metrics.base import SupervisedMetric

# point NLTK at our local data dir so it doesn't need system-wide installs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_nltk_data = os.path.join(project_root, "nltk_data")
if _nltk_data not in nltk.data.path:
    nltk.data.path.insert(0, _nltk_data)


class Rouge1(SupervisedMetric):
    __name__ = "Rouge1"
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0
        return self.scorer.score(correct_answer, answer)['rouge1'].fmeasure


class Rouge2(SupervisedMetric):
    __name__ = "Rouge2"
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0
        return self.scorer.score(correct_answer, answer)['rouge2'].fmeasure


class RougeL(SupervisedMetric):
    __name__ = "RougeL"
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer:
            return 0.0
        scores = self.scorer.score(correct_answer, answer)
        return scores['rougeLsum'].fmeasure


class Bleu(SupervisedMetric):
    __name__ = "Bleu"
    def __init__(self):
        self.metric = BLEU(effective_order=True)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer:
            return 0.0
        return self.metric.sentence_score(answer, [correct_answer]).score / 100.0


class Chrf(SupervisedMetric):
    __name__ = "ChrF"
    def __init__(self):
        self.metric = CHRF(word_order=0)  # char n-grams only

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0
        return self.metric.sentence_score(answer, [correct_answer]).score / 100.0


class ChrfPlus(SupervisedMetric):
    __name__ = "ChrFPlus"
    def __init__(self):
        self.metric = CHRF(word_order=2)  # ChrF++ uses word bigrams

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0
        return self.metric.sentence_score(answer, [correct_answer]).score / 100.0


class Ter(SupervisedMetric):
    __name__ = "Ter"
    def __init__(self):
        self.metric = TER()

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0
        score = self.metric.sentence_score(answer, [correct_answer]).score
        # TER is an error rate so we flip it into a similarity score
        return 1.0 - (score / 100.0)


class Meteor(SupervisedMetric):
    __name__ = "Meteor"

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer:
            return 0.0
        ans_tok = nltk.word_tokenize(answer)
        ref_tok = nltk.word_tokenize(correct_answer)
        return meteor_score([ref_tok], ans_tok)