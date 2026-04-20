from __future__ import annotations

import pandas as pd
from frozendict import frozendict

from models import SemanticPortrait
from parameters import IOParameters
from utils.constants import CSV_SEPARATORS
from utils.exceptions import BouzygesError
from utils.logger import LOGGER


class FileReader:
    """\
A class to read and parse files.
"""

    input_doctext = """\
<span>
    Input file must be a CSV file with at least the following columns:
    <ul>
        <li><code>vocab</code>: the vocabulary or code-system of the source term</li>
        <li><code>code</code>: the unique code for a source term</li>
        <li><code>term</code>: the term to be analyzed</li>
    </ul>
    The rest of the columns are optional; their values will be considered to be
    additional context for the term.
</span>
"""

    def __init__(self, read_parameters: IOParameters) -> None:
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.read_parameters = read_parameters
        self.content: pd.DataFrame | None = None

    def read(self) -> None:
        """\
Read the file
"""

        try:
            df = pd.read_csv(
                self.read_parameters.file,
                dtype=str,
                sep=CSV_SEPARATORS[self.read_parameters.sep],
                quotechar=self.read_parameters.quotechar,
            )
        except Exception as e:
            self.logger.error(
                f"Could not read {self.read_parameters.file}: {e}", exc_info=e
            )
            raise

        self.logger.info(
            f"Read {len(df)} rows from {self.read_parameters.file}"
        )
        self.content = df

    def parse(self) -> list[SemanticPortrait]:
        """\
Parse the file into a list of SemanticPortrait objects
"""
        if self.content is None:
            raise BouzygesError("No content to parse")

        return list(self.content.apply(self._portrait_from_row, axis=1))

    @staticmethod
    def _portrait_from_row(row: pd.Series) -> SemanticPortrait:
        metadata = frozendict(vocab=str(row["vocab"]), code=str(row["code"]))
        term = str(row["term"])
        context = [
            str(v)
            for k, v in row.to_dict().items()
            if k not in ["vocab", "code", "term"]
        ]
        return SemanticPortrait(
            term,
            context,
            metadata,
        )
