import pandas as pd
import json
import sys
import os

# add project root to path so we can import from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# golden/reference answers for each question - needed by reference-based metrics
GOLDEN_ANSWERS = {
    "In its loss function YoloV3 uses logistic regression with multilabel classification or Softmax over all class probabilities?": "YOLOv3 uses logistic regression with multilable classification. It employs independent logistic classifier for each class, rather than a softmax function. This approach is chosen because the authors found it to be sufficient for achieving good performance without the need for softmax normalization accross all classes. The loss function used is binary cross-entropy, suitable for multilabelclassification tasks",
    "When defining the reading comprehension task, the authors explain that they wish to estimate p(a|c, q). What would a model trained on this task do if the context \"c\" itself had factually incorrect information?": "The authors are training a reading comprehension model. Therefore, if the context \u201cc\u201d has incorrect information, the model is likely to answer based on the factually incorrect information itself. The authors clearly explain that the task their model is being built for and evaluated on is of identifying answers from a given text (i.e. comprehension) and not knowledge of global correctness.",
    "What baselines did they compare their model with?": "The passage states that the baseline approach they are comparing their model against is based on BIBREF20. This baseline utilizes a sequence-to-sequence model with attention for path generation (similar to BIBREF23 and BIBREF6) and depth-first search for path verification",
    "What datasets are used for training/testing models?": "The experiment dataset used for training and testing models comes from Microsoft Research (MSR) and consists of three domains: movie, taxi, and restaurant. The dataset is split into training, validation, and testing sets, with the exact count of dialogues per domain and split reported in Table TABREF11. The dataset provides annotated agent acts, which are used as targets in the experiment, and also includes information about the state at each turn obtained from MSR\u2019s dialogue management code and knowledge base.",
    "What kind of educational programs against smokeless tobacco are effective?": "While the provided text focuses on countermarketing efforts against youth smoking, it doesn\u2019t specifically address smokeless tobacco or the effectiveness of educational programs targeting it. To find information on effective programs against smokeless tobacco, you would need to consult sources that specifically address that topic"
}


def main():
    input_path = os.path.join(project_root, "questionnaire_input_2.csv")
    output_path = os.path.join(project_root, "benchmark_data.jsonl")

    print("Preparing benchmark data from questionnaire...")

    try:
        df = pd.read_csv(input_path, encoding='latin-1')
        print(f"  loaded {len(df)} examples from {input_path}")
    except FileNotFoundError:
        print(f"ERROR: couldn't find '{input_path}' - make sure it's in the project root")
        return

    # map each question to its golden answer
    df['correct_answer'] = df['question'].map(GOLDEN_ANSWERS)

    # pipeline expects 'answer' not 'generated_answer'
    df.rename(columns={'generated_answer': 'answer'}, inplace=True)

    benchmark_df = df[['question', 'context', 'answer', 'correct_answer']]

    # write out as jsonl
    with open(output_path, 'w') as f:
        for record in benchmark_df.to_dict('records'):
            f.write(json.dumps(record) + '\n')

    print(f"  wrote benchmark data to {output_path} ({len(benchmark_df)} rows)")
    print("Done.\n")


if __name__ == "__main__":
    main()