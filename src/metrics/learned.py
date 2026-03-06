import logging
import math
import torch

from src.metrics.base import SupervisedMetric, RAGMetric
from bleurt_pytorch import BleurtTokenizer, BleurtForSequenceClassification
from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# TF is only needed for BEM - don't crash if it's missing
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text  # registers custom ops for the preprocessor
    _TF_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not found - BEM metric won't be available")
    _TF_AVAILABLE = False


class Bleurt(SupervisedMetric):
    __name__ = "Bleurt"

    def __init__(self, checkpoint: str = "lucadiliello/BLEURT-20-D12"):
        self.tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
        self.model = BleurtForSequenceClassification.from_pretrained(checkpoint)
        self.model.eval()

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0
        with torch.no_grad():
            inputs = self.tokenizer([correct_answer], [answer], padding='longest', return_tensors='pt')
            logits = self.model(**inputs).logits.flatten().tolist()
        return logits[0]


class BartScore(SupervisedMetric):
    """Scores text by treating evaluation as a generation task (cross-entropy loss)."""
    __name__ = "BartScore"

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0

        inputs = self.tokenizer(answer, return_tensors="pt", truncation=True, padding=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(correct_answer, return_tensors="pt", truncation=True, padding=True)

        input_ids = inputs.input_ids.to(self.model.device)
        label_ids = labels.input_ids.to(self.model.device)

        loss = self.model(input_ids=input_ids, labels=label_ids).loss
        return math.exp(-loss.item())


class CrossEncoderSimilarity(SupervisedMetric):
    __name__ = "CrossEncoderSimilarity"

    def __init__(self, model_name: str = 'cross-encoder/stsb-roberta-large'):
        self.model = CrossEncoder(model_name)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        if not answer or not correct_answer: return 0.0
        return float(self.model.predict([(answer, correct_answer)]))


class BEM(RAGMetric):
    """BERT-based answer equivalence model (requires TensorFlow)."""
    __name__ = "BEM"

    def __init__(self):
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow/TF Hub/TF Text required for BEM")

        self.preprocessor = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        )
        self.bem_model = hub.KerasLayer(
            "https://tfhub.dev/google/answer_equivalence/bem/1"
        )

    def evaluate(self, question: str, answer: str, correct_answer: str, **kwargs) -> float:
        if not all([question, answer, correct_answer]): return 0.0

        inputs = {
            "query": tf.constant([question]),
            "reference": tf.constant([correct_answer]),
            "candidate": tf.constant([answer]),
        }
        encoded = self.preprocessor(inputs)
        logits = self.bem_model(encoded)
        prob = tf.nn.softmax(logits, axis=1).numpy()[0][0]
        return float(prob)