import json
import sys
import os

# some metrics (ragas etc.) expect this env var even if we don't use OpenAI directly.
# set a dummy value so they don't crash on init.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "not-needed"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation import EvaluationPipeline, RAGExample
from src.metrics.lexical import *
from src.metrics.semantic import *
from src.metrics.learned import *
from src.metrics.prompt_based import *


def load_benchmark(path):
    """Read benchmark examples from a jsonl file."""
    examples = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(RAGExample(**data))
    return examples


def main():
    input_file = os.path.join(project_root, "benchmark_data.jsonl")
    output_file = os.path.join(project_root, "automated_evaluation_results.jsonl")

    print("Running automated evaluation suite...")

    try:
        benchmark_data = load_benchmark(input_file)
        print(f"  {len(benchmark_data)} examples loaded")
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found. Run 01_prepare_benchmark_from_questionnaire.py first.")
        return

    # using smaller model variants to keep memory usage reasonable
    metrics = [
        # lexical
        Rouge1(), Rouge2(), RougeL(), Bleu(), Chrf(), ChrfPlus(), Ter(), Meteor(),
        # semantic
        SentenceSimilarity(), WordMoversSimilarity(), SentenceMoversSimilarity(),
        # learned
        BartScore(model_name="facebook/bart-base"),
        CrossEncoderSimilarity(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
        Bleurt(),
        # BEM(),  # disabled - TF dependency is heavy
        # prompt-based
        CustomLLMJudge(model_name="google/flan-t5-small"),
    ]

    pipeline = EvaluationPipeline(metrics=metrics)
    results = pipeline.run(benchmark_data)

    # flatten results into per-example records
    records = []
    for i, ex in enumerate(benchmark_data):
        rec = {
            "question": ex.question,
            "generated_answer": ex.answer,
        }
        for metric_name, scores in results.scores_raw.items():
            rec[metric_name] = scores[i] if (scores and i < len(scores)) else None
        records.append(rec)

    with open(output_file, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"  results saved to {output_file}")
    print("Done.\n")


if __name__ == "__main__":
    main()