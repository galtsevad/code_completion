import argparse
import json
import os


COLOR_MIDDLE = "\033[96m"
COLOR_GENERATED = "\033[92m"
CPLOR_END = "\033[0m"


def input_or_edit(sample: dict, key: str, val_type: type) -> None:
    """
    Input a value of a key in a sample or give a possibility to edit if exists
    """
    if sample.get(key, None) is not None:
        print("Previous value is:", sample[key])
        while True:
            try:
                user_input = input(key + ": Input new vaue or press enter to keep: ")
                if user_input != "":
                    sample[key] = val_type(user_input)
                break
            except ValueError:
                print("Incorrect value type, enter again")
    else:
        while True:
            try:
                user_input = input(key + ": ")
                sample[key] = val_type(user_input)
                break
            except ValueError:
                print("Incorrect value type, enter again: ")


def annotate(file_name: str, output: str) -> None:
    """
    For each sample display code with middle written in COLOR_MIDDLE,
    generation result (if exists) written in COLOR_GENERATED
    and with corresponding metrics.
    Annotations and labels should be inserted for each sample
    """
    with open(file_name, "r", encoding="utf-8") as jsonfile:
        dataset = json.load(jsonfile)

    for code in dataset:
        for sample in code["samples"]:
            os.system("cls" if os.name == "nt" else "clear")
            print(code["text"][: sample["middle_start"]], end="")
            print(COLOR_MIDDLE + sample["middle"] + CPLOR_END, end="")
            if sample.get("generation_result", None) is not None:
                print(COLOR_GENERATED + sample["generation_result"] + CPLOR_END)
            print(code["text"][sample["middle_end"]:])
            if sample.get("metrics", None) is not None:
                for metric, value in sample["metrics"].items():
                    print(metric, value, sep=": ")

            input_or_edit(sample, "annotations", str)
            input_or_edit(sample, "label", float)

    with open(output, "w", encoding="utf-8") as jsonfile:
        json.dump(dataset, jsonfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="output.json", type=str, help="Dataset file name")
    parser.add_argument("--output", default="annotated.json", type=str, help="Output file name")
    annotate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
