"""
Run model and compute metrics
"""

import argparse
import difflib
import json
import warnings
import evaluate
import Levenshtein
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_syntactic_correctness(code: str):
    """
    Return 1 if code is syntacricly correct
    """
    try:
        compile(code, "<string>", "exec")
        return 1
    except SyntaxError:
        return 0


def calculate_metrics(sample: dict, code: str) -> None:
    """
    Compute metrics:
    - chrf
    - exact_match
    - syntactic_correctness
    - levenshtein
    - lcs
    - normalized lcs
    - chrf++
    """
    chrf = evaluate.load("chrf")
    exact_match = evaluate.load("exact_match")
    sample["metrics"] = {}

    sample["metrics"]["chrf"] = chrf.compute(
        predictions=[sample["generation_result"]], references=[sample["middle"]]
    )
    sample["metrics"]["exact_match"] = exact_match.compute(
        predictions=[sample["generation_result"]], references=[sample["middle"]]
    )
    sample["metrics"]["syntactic_correctness"] = check_syntactic_correctness(
        code[:sample["middle_start"]] + sample["generation_result"] + code[sample["middle_end"]:]
    )
    sample["metrics"]["levenshtein"] = Levenshtein.ratio(
        sample["generation_result"], sample["middle"]
    )
    matcher = difflib.SequenceMatcher(a=sample["generation_result"], b=sample["middle"])
    sample["metrics"]["lcs"] = matcher.find_longest_match().size
    sample["metrics"]["lcs_normalized"] = (
        sample["metrics"]["lcs"] / len(sample["middle"]) if len(sample["middle"]) != 0 else 0
    )
    # chrf++ is chrf with word_order=2
    sample["metrics"]["chrf++"] = chrf.compute(
        predictions=[sample["generation_result"]], references=[sample["middle"]], word_order=2
    )


def run_model(model_id: str, file_name: str, output: str, max_new_tokens: int, device: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    eos = tokenizer.special_tokens_map["eos_token"]
    max_token_limit = model.config.max_position_embeddings

    with open(file_name, "r", encoding="utf-8") as jsonfile:
        dataset = json.load(jsonfile)

    for code in dataset:
        for sample in code["samples"]:
            # input to the model
            # <fim_prefix>prefix<fim_suffix>suffix<fim_middle>
            input_text = f'<fim_prefix>{code["text"][:sample["middle_start"]]}\
                <fim_suffix>{code["text"][sample["middle_end"]:]}<fim_middle>'
            inputs = tokenizer(input_text, return_tensors="pt", return_token_type_ids=False).to(
                device
            )
            # check that input and expected output fits in the model
            if inputs["input_ids"].shape[-1] > max_token_limit - max_new_tokens:
                warnings.warn("The input is too big")
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generation = [tokenizer.decode(tensor) for tensor in outputs]
            # select generated part
            result = generation[0].split("<fim_middle>")[-1].split(eos)[0]
            sample["generation_result"] = result
            calculate_metrics(sample, code["text"])

    with open(output, "w", encoding="utf-8") as jsonfile:
        json.dump(dataset, jsonfile)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", default="bigcode/tiny_starcoder_py", type=str, help="HuggingFace model id"
    )
    parser.add_argument("--file_name", default="dataset.json", type=str, help="Dataset file name")
    parser.add_argument("--output", default="output.json", type=str, help="Output file name")
    parser.add_argument(
        "--max_new_tokens", default=25, type=int, help="Maximum number of new tokens for generation"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to run the model: cpu or cuda"
    )
    run_model(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
