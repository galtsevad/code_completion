# code_completion
JetBrains AI Evaluation Internship Test Task

The detailed report is in the file ___report.pdf___.

`python3 annotate.py --file_name resulting_dataset.json` can be used to visualize the dataset.

# Dataset structure:
Dataset is saved in json format: list of dictionaries for each file that was used for dataset creation:

`[{"filename": str, "text": str, "samples": list}, …]`

Each sample is a dictionary with start position of middle part in the code, end position of middle part and a middle part itself (even though it is not really needed, because it can be extracted from “text” field of corresponding file entry):

`"samples": [{"middle": str, "middle_start": int, "middle_end": int}, …]`

*After running a model*, fields “generation_result” and “metrics” with a dictionary of metrics are added.

*After annotating*, fields “annotations” with text annotations and “label” with float label are added.

Example of accessing chrf++ metric of first sample from first file:

`dataset[0]["samples"][0]["metrics"]["chrf++"]`
