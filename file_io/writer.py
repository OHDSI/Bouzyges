from __future__ import annotations

import datetime
import json
import logging
import os
from typing import (
    Callable,
    Iterable,
    Coroutine,
)

import pandas as pd
from models import WrappedResult
from parameters import IOParameters
from snomed import SnowstormAPI
from utils.constants import CSV_SEPARATORS
from utils.exceptions import BouzygesError
from utils.logger import LOGGER
from utils.types import (
    SCTID,
    Json,
    OutFormat,
)


class FileWriter:
    """\
A class to write resulting files.
"""

    def __init__(
        self,
        path: str,
        write_parameters: IOParameters,
        append: bool,
        format: OutFormat = "SCG",
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.logger = logger.getChild(self.__class__.__name__)
        self.path = path
        self.content: pd.DataFrame | Json | None = None
        self.write_parameters = write_parameters
        self.append = append

        self.write_chosen: Callable[
            [Iterable[WrappedResult], SnowstormAPI], Coroutine
        ]
        match format:
            case "SCG":
                self.write_chosen = self.to_snomed_compositional_grammar
            case "CRS":
                self.write_chosen = self.to_concept_relationship_stage
            case "JSON":
                self.write_chosen = self.to_json
            case _:
                raise NotImplementedError

    @staticmethod
    def get_formats() -> list[OutFormat]:
        return ["SCG", "CRS", "JSON"]

    def _write_csv(self) -> None:
        if self.content is None:
            raise BouzygesError("No content to write")

        if not isinstance(self.content, pd.DataFrame):
            raise BouzygesError("Content is not a DataFrame")

        self.logger.debug("Content ready")

        self.logger.info(f"Writing to {self.path}")

        try:
            self.content.to_csv(
                self.path,
                index=False,
                sep=CSV_SEPARATORS[self.write_parameters.sep],
                quotechar=self.write_parameters.quotechar,
                quoting=self.write_parameters.quoting,
                na_rep="",
                mode="a" if self.append else "w",
            )
        except Exception as e:
            self.logger.error(
                f"Could not write to {self.path}: {e}", exc_info=e
            )
            raise

        self.logger.info(f"Written to {self.path}")

    async def to_snomed_compositional_grammar(
        self, results: Iterable[WrappedResult], snowstorm: SnowstormAPI
    ) -> None:
        """\
Write the results of term evaluation as a table of SNOMED CT Post-Coordinated
Expressions.

Does not normalize nor verify the expressions, nor considers grouping rules;
some tools like CSIRO Ontoserver can do that. Unfortunately, normalization rules
are not formally defined.
"""
        self.logger.debug("Writing to SCG format")

        # TODO: use Snowstorm for annotations
        _ = snowstorm

        dicts = []
        for result in results:
            portrait, map_ = result.portrait, result.name_map
            row = {}
            row["term"] = portrait.source_term
            row.update(portrait.attributes)
            row["scg"] = portrait.to_scg()
            dicts.append(row)
            row["ancestors_json"] = json.dumps(
                [
                    {"conceptId": k, "pt": map_.get(k, "Unknown")}
                    for k in portrait.ancestor_anchors
                ]
            )
            row["ancestors_scg"] = "<<< " + " + ".join(
                f"{k} |{map_.get(k, 'Unknown')}|"
                for k in portrait.ancestor_anchors
            )
            self.content = pd.DataFrame(dicts)
        self._write_csv()

    async def to_concept_relationship_stage(
        self, results: Iterable[WrappedResult], snowstorm: SnowstormAPI
    ) -> None:
        """\
Write the results of term evaluation as a table in format of of OMOP CDM
concept_relationship table.

Warning: This is a very naive implementation and does not consider the state and
structure of SNOMED vocabulary in OMOP CDM. This will not respect actual
standard status of the concepts, version incompatibility, etc. Post-processing
WILL be required.

This will also not check for duplicates in `code` and `vocab` columns, nor any
other constraints.
"""
        self.logger.debug("Writing to CONCEPT_RELATIONSHIP_STAGE format")

        # TODO: use Snowstorm for annotations
        _ = snowstorm

        dicts = []
        today = datetime.date.today().strftime("%Y-%m-%d")
        for result in results:
            portrait = result.portrait
            if not portrait.ancestor_anchors:
                self.logger.warning(
                    f"No ancestors found for {portrait.source_term}"
                )
                continue

            metadata = portrait.metadata
            if not ("code" in metadata and "vocab" in metadata):
                self.logger.warning(
                    f"Crucial metadata missing for {portrait.source_term}!"
                )
                continue

            for ancestor in portrait.ancestor_anchors:
                row = {
                    "concept_code_1": metadata["code"],
                    "vocabulary_id_1": metadata["vocab"],
                    "concept_code_2": ancestor,
                    "vocabulary_id_2": "SNOMED",
                    "relationship_id": "Is a",
                    "valid_start_date": today,
                    "valid_end_date": "2099-12-31",
                    "invalid_reason": None,
                }
                dicts.append(row)
        self.content = pd.DataFrame(dicts)
        self._write_csv()

    async def to_json(
        self, results: Iterable[WrappedResult], snowstorm: SnowstormAPI
    ) -> None:
        """\
Write the results of term evaluation as a JSON file.

JSON schema:
    {
        "items": [
            {
                "term": "Pyogenic abscess of liver",
                "attributes": [
                    {
                        "attribute": {
                            "id": 363698007,
                            "pt": "Finding site"
                        },
                        "value": {
                            "id": 10200004,
                            "pt": "Liver structure"
                        }
                    },
                    ...
                ],
                "proximal_ancestors": [
                    {
                        "conceptId": 64572001,
                        "pt": "Disease"
                    },
                    ...
                ],
                "scg": "<<<64572001:363698007=10200004",
                "metadata": { ... }
            },
            ...
        ]
    }
"""
        self.logger.debug("Writing to JSON format")
        dicts = []
        annotations: dict[SCTID, str] = {}

        for result in results:
            portrait, anchors = result.portrait, result.name_map

            # Annotate attributes
            concepts = {
                *portrait.attributes.keys(),
                *portrait.attributes.values(),
            }
            new_concepts = concepts - set(annotations)
            new_annotations = await snowstorm.get_concepts(new_concepts)
            annotations.update({k: v.pt for k, v in new_annotations.items()})

            attributes = []
            for k_id, v_id in portrait.attributes.items():
                attributes.append(
                    {
                        "attribute": {"id": k_id, "pt": annotations[k_id]},
                        "value": {"id": v_id, "pt": annotations[v_id]},
                    }
                )

            row = {
                "term": portrait.source_term,
                "attributes": attributes,
                "proximal_ancestors": [
                    {"conceptId": k, "pt": v} for k, v in anchors.items()
                ],
                "scg": portrait.to_scg(),
                "metadata": dict(portrait.metadata),
            }
            dicts.append(row)

        # Get existing content
        existing = []
        if self.append:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    existing = json.load(f)["items"]
        self.logger.debug(f"Existing content of length {len(existing)} loaded")

        existing.extend(dicts)
        self.content = {"items": existing}
        self.logger.debug(f"Total content length: {len(existing)}")

        with open(self.path, "w") as f:
            try:
                json.dump(self.content, f, indent=2)
            except Exception as e:
                self.logger.error(
                    f"Could not write to {self.path}: {e}", exc_info=e
                )
                raise

        self.logger.info(f"Written to {self.path}")
