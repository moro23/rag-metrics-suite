import pandas as pd
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    questionnaire_file = os.path.join(project_root, "questionnaire_input_2.csv")
    human_responses_file = os.path.join(project_root, "human_annotated_results_2.csv")
    auto_scores_file = os.path.join(project_root, "automated_evaluation_results.jsonl")
    output_file = os.path.join(project_root, "analysis_ready_data.csv")

    print("Processing human annotations and merging with automated scores...")

    try:
        df_base = pd.read_csv(questionnaire_file)
        df_base['generated_answer'] = df_base['generated_answer'].str.strip()
        print(f"  {len(df_base)} questions from questionnaire")

        df_raw_human = pd.read_csv(human_responses_file)

        # score columns start after the 9 consent/demographic columns
        score_cols = df_raw_human.columns[9:]

        df_scores = df_raw_human.dropna(subset=score_cols).copy()
        df_scores[score_cols] = df_scores[score_cols].apply(pd.to_numeric, errors='coerce')
        print(f"  {len(df_scores)} completed survey responses")

        # collect per-answer scores across all respondents
        all_scores = [[] for _ in range(len(df_base))]
        for _, row in df_scores.iterrows():
            for i, col in enumerate(score_cols):
                if i < len(all_scores):
                    all_scores[i].append(row[col])

        df_base['avg_human_score'] = [pd.Series(s).mean() for s in all_scores]
        print("  computed average human scores per answer")

        # load automated metric results
        df_auto = pd.read_json(auto_scores_file, lines=True)
        df_auto['generated_answer'] = df_auto['generated_answer'].str.strip()
        print(f"  {len(df_auto)} automated metric results loaded")

        # merge on answer text
        df_merged = pd.merge(
            df_base,
            df_auto.drop(columns=['question']),
            on="generated_answer",
            how="inner"
        )
        print(f"  merged dataset: {len(df_merged)} rows")

        df_merged.to_csv(output_file, index=False)
        print(f"  saved to {output_file}")
        print("Done.\n")

    except FileNotFoundError as e:
        print(f"\nERROR: missing file - {e.filename}")
        print("Make sure these exist in the project root:")
        print("  - questionnaire_input_2.csv")
        print("  - human_annotated_results_2.csv")
        print("  - automated_evaluation_results.jsonl (run script 03 first)")


if __name__ == "__main__":
    main()