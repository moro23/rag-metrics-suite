import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    analysis_file = os.path.join(project_root, "analysis_ready_data.csv")
    heatmap_path = os.path.join(project_root, "final_correlation_heatmap.png")

    print("Analyzing metric correlations...")

    try:
        df = pd.read_csv(analysis_file)
        print(f"  {len(df)} records loaded")
    except FileNotFoundError:
        print(f"ERROR: {analysis_file} not found. Run 02_process_and_merge_data.py first.")
        return

    # only keep numeric score columns for correlation
    exclude = {'example_id', 'question', 'context', 'generated_answer'}
    score_cols = [c for c in df.columns if c not in exclude]
    corr = df[score_cols].corr(method='spearman')

    # print how each metric correlates with human scores
    print("\nCorrelation with avg_human_score (Spearman):")
    human_corr = corr[['avg_human_score']].drop('avg_human_score')
    print(human_corr.sort_values('avg_human_score', ascending=False).to_string())

    # plot the full correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Spearman Correlation: Automated Metrics vs. Human Judgment", fontsize=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig.savefig(heatmap_path, dpi=150)
    print(f"\n  heatmap saved to {heatmap_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()