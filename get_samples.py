import argparse
import json
import random
import numpy as np


def get_suffix_position(line: str, start: int) -> int:
    """
    Get suffix start position from a line.
    If completing the code inside brackets then return the
    position of the closing bracket.
    Else return the end of the line.
    """
    brackets = 0
    for i, char in enumerate(line[start:]):
        if char in "([{":
            brackets += 1
        elif char in ")]}":
            brackets -= 1
        if brackets < 0:
            return i + start
    return len(line) - 1   # -1 not to sample \n


def get_samples(file_names: list[str], output: str, seed: int, num_samples: int) -> None:
    """
    Split given files into three parts simulating the user cursor position:
    the prefix - code before the cursor, the suffix - code after the cursor,
    and the middle - code that is missing and we assume should be typed next.
    """
    dataset = []
    random.seed(seed)
    for filename in file_names:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
        char_num = np.array([len(line) for line in lines])  # lengts of each line
        line_end_pos = char_num.cumsum()
        sampled_idx = random.sample(
            range(line_end_pos[-1]),  # starting positions of samples
            min(num_samples, line_end_pos[-1]),
        )
        sampled_lines = np.searchsorted(line_end_pos, sampled_idx, side="right")
        samples = []
        for line, pos in zip(sampled_lines, sampled_idx):
            line_pos = len(lines[line]) + pos - line_end_pos[line]  # position of a sample in a line
            suffix_pos = get_suffix_position(lines[line], int(line_pos))
            samples.append(
                {
                    "middle": "".join(lines[line][line_pos:suffix_pos]),
                    "middle_start": pos,
                    "middle_end": (
                        int(line_end_pos[line - 1]) + suffix_pos if line > 0 else suffix_pos
                    ),
                }
            )
        dataset.append({"filename": filename, "text": "".join(lines), "samples": samples})
    with open(output, "w", encoding="utf-8") as jsonfile:
        json.dump(dataset, jsonfile)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_names",
        default=["get_samples.py", "annotate.py"],
        type=str,
        nargs="+",
        help="List of file names",
    )
    parser.add_argument("--output", default="dataset.json", type=str, help="Output file name")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num_samples", default=25, type=int, help="Number of samples per file")
    get_samples_args = parser.parse_args()
    get_samples(**vars(get_samples_args))


if __name__ == "__main__":
    main()
