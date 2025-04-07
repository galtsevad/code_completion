import argparse
from collections import defaultdict
import json
from scipy import stats


def correlation(file_name: str) -> None:
    """
    Calculate Spearman correlation between labels and metrics
    """
    with open(file_name, "r", encoding="utf-8") as jsonfile:
        dataset = json.load(jsonfile)

    metrics = defaultdict(list)
    labels = []
    for code in dataset:
        metrics["exact_match"].extend(
            [i["metrics"]["exact_match"]["exact_match"] for i in code["samples"]]
        )
        for m in ["chrf", "chrf++"]:
            metrics[m].extend([i["metrics"][m]["score"] for i in code["samples"]])
        for m in ["syntactic_correctness", "levenshtein", "lcs", "lcs_normalized"]:
            metrics[m].extend([i["metrics"][m] for i in code["samples"]])
        labels.extend([i["label"] for i in code["samples"]])

    print("Pearson")
    for metric, values in metrics.items():
        res = stats.pearsonr(values, labels)
        print(f"{metric:<25}\tstatistic={res.statistic:.2f},   p_value={res.pvalue:.2e}")
    print()
    print("Spearman")
    for metric, values in metrics.items():
        res = stats.spearmanr(values, labels)
        print(f"{metric:<25}\tstatistic={res.statistic:.2f},   p_value={res.pvalue:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="annotated.json", type=str, help="Dataset file name")
    correlation(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
