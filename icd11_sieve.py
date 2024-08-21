"""\
A small script to sample ICD-11 codes from the download file.

The ICD-11 download file is available at:
https://icd.who.int/dev11/downloads
"""

import csv
import re

import pandas as pd

SAMPLE_SIZE = 45
FILE_PATH = "LinearizationMiniOutput-MMS-en.txt"


def clean_term(term):
    term = re.sub(r"^(- )+", "", term)
    term = term.strip()
    return term


def main():
    icd11 = pd.read_csv(FILE_PATH, sep="\t", dtype=str)
    icd11 = icd11[icd11["isLeaf"] == "True"]
    # Exclude the "X" codes, which are not useful for our purposes
    icd11 = icd11[~icd11["Code"].str.startswith("X", na=False)]
    icd11["code"] = icd11["Code"]
    icd11["term"] = icd11["Title"].apply(clean_term)
    icd11["vocab"] = "ICD-11"
    icd11 = icd11[["code", "term", "vocab"]]
    icd11 = icd11.sample(SAMPLE_SIZE)
    icd11.to_csv(
        "Test.csv", index=False, sep="\t", quotechar='"', quoting=csv.QUOTE_ALL
    )


if __name__ == "__main__":
    main()
