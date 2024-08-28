from __future__ import annotations

import asyncio
import cProfile
import csv
import datetime
import itertools
import json
import logging
import os
import pstats
import re
import sqlite3
import sys
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    Iterable,
    Literal,
    Mapping,
    Self,
    TypeAlias,
    TypeVar,
)

import openai
import pandas as pd
import pydantic
import requests
import tenacity
import webbrowser
from frozendict import frozendict
import qasync
from PyQt6 import QtCore, QtGui, QtWidgets

# Optional imports
## dotenv
try:
    import dotenv
except ImportError:
    logging.info(
        "python-dotenv package not installed. "
        "Relying on explicit environment variables"
    )
    dotenv = None

## tiktoken
try:
    import tiktoken
except ImportError:
    logging.warning(
        "tiktoken package is not installed. "
        "Will not be able to use track token usage"
    )
    tiktoken = None


# Boilerplate
## Typing
class BranchPath(str):
    """Snowstorm working branch"""


class SCTID(int):
    """SNOMED CT identifier"""


class SCTDescription(str):
    """Any of valid SNOMED CT descriptions

    Prefer PT for LLMs and FSN for humans.
    """


class ECLExpression(str):
    """Expression Constraint Language expression"""


class SCGExpression(str):
    """SNOMED CT Compositional Grammar expression"""


class EscapeHatch(object):
    """\
"Escape hatch" sentinel type for prompters

Escape hatch is provided to an LLM agent to be able to choose nothing rather
than hallucinating an answer. Will have just one singleton instance.
"""

    WORD: SCTDescription = SCTDescription("[NONE]")


class BooleanAnswer(str):
    """\
Boolean answer constants for prompters for yes/no questions
"""

    YES = SCTDescription("[AYE]")
    NO = SCTDescription("[NAY]")

    def __new__(cls, value: bool):
        return cls.YES if value else cls.NO


PrompterOption: TypeAlias = Literal["human", "openai", "azure"]
JsonPrimitive: TypeAlias = int | float | str | bool | None
Json: TypeAlias = dict[str, "Json"] | list["Json"] | JsonPrimitive
JsonDict: TypeAlias = dict[str, Json]
OpenAIPromptRole: TypeAlias = Literal["user", "system", "assisstant"]
OpenAIMessages: TypeAlias = tuple[frozendict[OpenAIPromptRole, str]]
OutFormat: TypeAlias = Literal["SCG", "CRS", "JSON"]
T = TypeVar("T")
Url = str

## Logging
LOGGER = logging.getLogger("Bouzyges")
logging.basicConfig(level=logging.INFO)
LOGGER.info("Logging started")
# Default handler and formatter
LOGGER.handlers.clear()
_stdout_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] "
    "%(message)s"
)
_stdout_handler.setFormatter(_formatter)
LOGGER.addHandler(_stdout_handler)
LOGGER.info("Logging configured")

## Parameters
# Load environment variables for API access
if dotenv is not None:
    LOGGER.info("Loading environment variables from .env")
    if os.path.exists(".env"):
        dotenv.load_dotenv()
    else:
        LOGGER.warning("No .env file found")

DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_PROMPTERS: dict[PrompterOption, str] = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "human": "Human",
}
SEPARATORS: list[str] = [",", ";", "Tab"]
QUOTECHARS: list[str] = ['"', "'"]
QUOTING_POLICY: dict[int, str] = {
    csv.QUOTE_MINIMAL: "Quote minimal",
    csv.QUOTE_ALL: "Quote all",
    csv.QUOTE_NONNUMERIC: "Quote string",
    csv.QUOTE_NONE: "No quoting",
}


class ProfilingParameters(pydantic.BaseModel):
    """\
Parameters to control profiling of the program.
"""

    enabled: bool
    stop_profiling_after_seconds: int | None


class LoggingParameters(pydantic.BaseModel):
    """\
Parameters to control logging.
"""

    log_to_file: bool
    logging_level: int

    def update(self, level):
        self.logging_level = level
        LOGGER.setLevel(level=self.logging_level)


class APIParameters(pydantic.BaseModel):
    """\
Parameters to control the interface to Snowstorm, cache and LLMs.
"""

    prompter: PrompterOption
    snowstorm_url: Url
    llm_model_id: str
    cache_db: str | None


class EnvironmentParameters(pydantic.BaseModel):
    """\
Parameters that reflect the environment variables.
"""

    OPENAI_API_KEY: str | None = None
    AZURE_API_KEY: str | None = None
    AZURE_API_ENDPOINT: str | None = None

    def fill_from_env(self) -> None:
        for env in self.model_fields:
            env_value = os.getenv(env)
            setattr(self, env, env_value or None)


class IOParameters(pydantic.BaseModel):
    """\
Parameters for reading and writing CSV file data.
"""

    file: str
    sep: str
    quotechar: str
    quoting: int


class IOParametersWidget(QtWidgets.QWidget):
    def __init__(self, par: IOParameters, name: str, *args, **kwargs) -> None:
        # Add pydantic fields
        super().__init__(*args, **kwargs)

        self.logger = LOGGER.getChild(name)
        self.parameters: IOParameters = par
        self._populate_layout()
        self.set_values()

    def _populate_layout(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        self.sep_cb = QtWidgets.QComboBox()
        self.sep_cb.addItems(map(lambda s: "Separator: " + s, SEPARATORS))
        self.sep_cb.currentIndexChanged.connect(self.separator_changed)
        layout.addWidget(self.sep_cb)
        self.quoting_cb = QtWidgets.QComboBox()
        self.quoting_cb.addItems(QUOTING_POLICY.values())
        self.quoting_cb.currentIndexChanged.connect(self.quoting_policy_changed)
        layout.addWidget(self.quoting_cb)
        qchar_label = QtWidgets.QLabel("Quote character:")
        layout.addWidget(qchar_label)
        self.qc_edit = QtWidgets.QLineEdit()
        self.qc_edit.setPlaceholderText('"')
        self.qc_edit.setMaximumWidth(30)
        self.qc_edit.textChanged.connect(self.quote_char_changed)
        layout.addWidget(self.qc_edit)
        spacer = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        layout.addItem(spacer)
        self.setLayout(layout)

    def separator_changed(self, index) -> None:
        self.parameters.sep = SEPARATORS[index]
        self.logger.debug(f"Separator changed to: {SEPARATORS[index]}")

    def quoting_policy_changed(self, index) -> None:
        self.parameters.quoting = list(QUOTING_POLICY)[index]
        self.logger.debug(
            f"Quoting policy changed to: {self.parameters.quoting}:"
            f"{QUOTING_POLICY[self.parameters.quoting]}"
        )
        self.qc_edit.setEnabled(self.parameters.quoting != csv.QUOTE_NONE)

    def quote_char_changed(self, text) -> None:
        self.parameters.quotechar = text
        self.logger.debug(f"Quote character changed to: {repr(text)}")

    def update_file(
        self, file: str, update: Callable[[str], None] | None = None
    ) -> None:
        self.parameters.file = file
        if update:
            update(file)
        self.logger.debug(f"Input file set to: {file}")

    def set_values(self) -> None:
        self.sep_cb.setCurrentIndex(SEPARATORS.index(self.parameters.sep))
        self.quoting_cb.setCurrentIndex(
            list(QUOTING_POLICY).index(self.parameters.quoting)
        )
        self.qc_edit.setText(self.parameters.quotechar)
        self.qc_edit.setEnabled(self.parameters.quoting != csv.QUOTE_NONE)


class RunParameters(pydantic.BaseModel):
    """\
Parameters for the run of the program.
"""

    api: APIParameters
    env: EnvironmentParameters = pydantic.Field(
        exclude=True, default_factory=EnvironmentParameters
    )
    log: LoggingParameters
    prof: ProfilingParameters
    read: IOParameters
    write: IOParameters
    format: OutFormat
    out_dir: str = pydantic.Field(default_factory=os.getcwd)

    @classmethod
    def from_file(cls, file: str) -> RunParameters:
        with open(file, "r") as f:
            json_data = json.load(f)
        params = RunParameters(**json_data)
        # Environment variables are not stored in JSON
        params.env.fill_from_env()
        return params

    def update(self, json_data: dict) -> None:
        self.__init__(**json_data)
        self.log.update(self.log.logging_level)

    def save(self, file: str) -> None:
        with open(file, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


PARAMS = RunParameters.from_file("default_config.json")
LOGGER.info(f"Parameters loaded: {json.dumps(PARAMS.model_dump(), indent=2)}")

## Logic constants
### MRCM
MRCM_DOMAIN_REFERENCE_SET_ECL = ECLExpression("<<723589008")
WHITELISTED_SUPERTYPES: set[SCTID] = {
    # Only limit to well-modeled supertypes for now
    SCTID(404684003),  # Clinical finding
    SCTID(71388002),  # Procedure
}

### Escape hatch sentinel
NULL_ANSWER = EscapeHatch()


### "Is a" relationships
IS_A = SCTID(116680003)


### SNOMED root concept
ROOT_CONCEPT = SCTID(138875005)

### Temporary substitute for reading from a file
TERMS = ["Pyogenic abscess of liver", "Invasive lobular carcinoma of breast"]


## Dataclasses
### SNOMED modelling
@dataclass(frozen=True, slots=True)
class AttributeRelationship:
    """Represents a SNOMED CT attribute-value pair relationship."""

    attribute: SCTID
    value: SCTID


@dataclass(frozen=True, slots=True)
class AttributeGroup:
    """Represents a SNOMED CT attribute group."""

    relationships: frozenset[AttributeRelationship]


@dataclass(frozen=True, slots=True)
class Concept:
    """Represents a SNOMED CT concept."""

    sctid: SCTID
    pt: SCTDescription
    fsn: SCTDescription
    groups: frozenset[AttributeGroup]
    ungrouped: frozenset[AttributeRelationship]
    defined: bool

    @classmethod
    def from_rela_json(cls, relationship_data: list[dict]):
        # Get concept proper info
        # Everyone is expected to have at least an 'Is a' relationship
        # Except root, but encountering it is an error
        if len(relationship_data) == 0:
            raise ValueError("No relationships found; could it be a root?")

        specimen = relationship_data[0]["source"]
        sctid = SCTID(specimen["conceptId"])
        pt = SCTDescription(specimen["pt"]["term"])
        fsn = SCTDescription(specimen["fsn"]["term"])
        defined = specimen["definitionStatus"] == "FULLY_DEFINED"

        # Get relationships
        groups: dict[int, set[AttributeRelationship]] = {}
        ungrouped: set[AttributeRelationship] = set()
        for relationship in relationship_data:
            attribute = SCTID(relationship["type"]["conceptId"])
            # Skip 'Is a' relationships here
            if attribute == IS_A:
                continue
            value = SCTID(relationship["target"]["conceptId"])
            group = relationship["groupId"]
            rel = AttributeRelationship(attribute, value)
            if group == 0:
                ungrouped.add(rel)
            else:
                groups.setdefault(group, set()).add(rel)

        return cls(
            sctid=sctid,
            pt=pt,
            fsn=fsn,
            groups=frozenset(
                AttributeGroup(frozenset(relationships))
                for relationships in groups.values()
            ),
            ungrouped=frozenset(ungrouped),
            defined=defined,
        )

    @classmethod
    def from_json(cls, json_data: dict):
        groups: dict[int, set[AttributeRelationship]] = {}
        ungrouped: set[AttributeRelationship] = set()
        for relationship in json_data["relationships"]:
            if not relationship["active"]:
                continue

            attribute = SCTID(relationship["type"]["conceptId"])
            # Skip 'Is a' relationships here
            if attribute == IS_A:
                continue
            value = SCTID(relationship["target"]["conceptId"])
            rel = AttributeRelationship(attribute, value)
            if (group := relationship["groupId"]) == 0:
                ungrouped.add(rel)
            else:
                groups.setdefault(group, set()).add(rel)

        match json_data["definitionStatus"]:
            case "FULLY_DEFINED":
                defined = True
            case "PRIMITIVE":
                defined = False
            case _:
                raise (ValueError("Unknown definition status"))

        return cls(
            sctid=SCTID(json_data["conceptId"]),
            pt=SCTDescription(json_data["pt"]["term"]),
            fsn=SCTDescription(json_data["fsn"]["term"]),
            groups=frozenset(
                AttributeGroup(frozenset(relationships))
                for relationships in groups.values()
            ),
            ungrouped=frozenset(ungrouped),
            defined=defined,
        )


@dataclass(frozen=True, slots=True)
class MRCMDomainRefsetEntry:
    """\
Represents an entry in the MRCM domain reference set.

Used mainly to obtain an entry ancestor anchor for a semantic portrait. Contains
more useful information for domain modelling, but it is not yet well explored.
"""

    # Concept properties
    sctid: SCTID
    term: SCTDescription

    # Additional fields
    domain_constraint: ECLExpression
    guide_link: Url  # For eventual RAG connection
    # Currently uses unparseable extension of ECL grammar,
    # but one day we will use it
    domain_template: ECLExpression
    parent_domain: ECLExpression | None = None
    proximal_primitive_refinement: ECLExpression | None = None

    @classmethod
    def from_json(cls, json_data: dict):
        af = json_data["additionalFields"]
        dom = af.get("parentDomain")
        prf = af.get("proximalPrimitiveRefinement")

        return cls(
            sctid=SCTID(json_data["referencedComponent"]["conceptId"]),
            term=SCTDescription(json_data["referencedComponent"]["pt"]["term"]),
            domain_template=ECLExpression(
                # Use precoordination: stricter
                af["domainTemplateForPrecoordination"]
            ),
            domain_constraint=ECLExpression(af["domainConstraint"]),
            guide_link=Url(af["guideURL"]),
            parent_domain=ECLExpression(dom) if dom else None,
            proximal_primitive_refinement=ECLExpression(prf) if prf else None,
        )


# Solving for presence vs. absence of attributes is way, way easier than solving
# quantitative problems. For now, Cardinality will likely be ignored.
@dataclass(frozen=True, slots=True)
class Cardinality:
    """\
Represents a cardinality of an attribute in a group or a definition.
"""

    min: int
    max: None | int


@dataclass(frozen=True, slots=True)
class AttributeDomain:
    """\
Represents a domain of applications of an attribute.
"""

    sctid: SCTID
    pt: SCTDescription
    domain_id: SCTID
    grouped: bool
    cardinality: Cardinality
    in_group_cardinality: Cardinality


@dataclass(frozen=True, slots=True)
class AttributeRange:
    """\
Represents a range of values of an attribute.
"""

    sctid: SCTID
    pt: SCTDescription
    range_constraint: ECLExpression
    attribute_rule: ECLExpression
    contentType: str


@dataclass(frozen=True, slots=True)
class AttributeConstraints:
    """\
Represents information about an attribute obtained from a known set of parent
concepts.
"""

    sctid: SCTID
    pt: SCTDescription
    attribute_domain: Iterable[AttributeDomain]
    # May actually not be required
    # because /mrcm/{branch}/attribute-values/ exists
    attribute_range: Iterable[AttributeRange]

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            sctid=SCTID(json_data["conceptId"]),
            pt=SCTDescription(json_data["pt"]["term"]),
            attribute_domain=[
                AttributeDomain(
                    sctid=SCTID(json_data["conceptId"]),
                    pt=SCTDescription(json_data["pt"]["term"]),
                    domain_id=SCTID(ad["domainId"]),
                    grouped=ad["grouped"],
                    cardinality=Cardinality(
                        ad["attributeCardinality"]["min"],
                        ad["attributeCardinality"].get("max"),
                    ),
                    in_group_cardinality=Cardinality(
                        ad["attributeInGroupCardinality"]["min"],
                        ad["attributeInGroupCardinality"].get("max"),
                    ),
                )
                for ad in json_data["attributeDomain"]
            ],
            attribute_range=[
                AttributeRange(
                    sctid=SCTID(json_data["conceptId"]),
                    pt=SCTDescription(json_data["pt"]["term"]),
                    range_constraint=ECLExpression(ar["rangeConstraint"]),
                    attribute_rule=ECLExpression(ar["attributeRule"]),
                    contentType=ar["contentType"],
                )
                for ar in json_data["attributeRange"]
            ],
        )


### Mutable portrait
@dataclass
class SemanticPortrait:
    """\
Represents an interactively built semantic portrait of a source concept."""

    def __init__(
        self,
        term: str,
        context: Iterable[str] | None = None,
        metadata: frozendict[str, JsonPrimitive] = frozendict(),
    ) -> None:
        self.source_term: str = term
        self.context: Iterable[str] | None = context
        self.ancestor_anchors: set[SCTID] = set()

        self.unchecked_attributes: set[SCTID] = set()
        self.attributes: dict[SCTID, SCTID] = {}
        self.rejected_attributes: set[SCTID] = set()
        self.rejected_supertypes: set[SCTID] = set()

        self.relevant_constraints: dict[SCTID, AttributeConstraints] = {}

        self.metadata: frozendict[str, JsonPrimitive] = metadata

    def to_scg(self: SemanticPortrait) -> SCGExpression:
        """\
Convert a SemanticPortrait to a SNOMED CT Post-Coordinated Expression
"""
        # Always Subtype
        prefix = "<<<"
        focus_concepts = "+".join(str(a) for a in self.ancestor_anchors)
        attributes = ",".join(f"{a}={v}" for a, v in self.attributes.items())
        return SCGExpression(
            prefix
            + (focus_concepts or str(ROOT_CONCEPT))
            + ":" * bool(attributes)
            + attributes
        )


class WrappedResult:
    """\
Represents a completed semantic portrait with additional metadata.
"""

    def __init__(
        self,
        portrait: SemanticPortrait,
        name_map: Mapping[SCTID, SCTDescription],
    ) -> None:
        self.portrait: SemanticPortrait = portrait
        self.name_map: Mapping[SCTID, SCTDescription] = name_map


## Exceptions
class ProfileMark(Exception):
    """Interrupts flow of the program at arbitrary point for profiling"""


class BouzygesError(Exception):
    """Base class for Bouzyges errors."""


class SnowstormAPIError(Exception):
    """Raised when the Snowstorm API returns a bad response."""


class SnowstormRequestError(SnowstormAPIError):
    """Raised when the Snowstorm API returns a non-200 response"""

    def __init__(self, text, response, *_):
        super().__init__(text)
        self.response = response

    @classmethod
    def from_response(cls, response):
        LOGGER.error(
            f"Request: {response.request.method}, {response.request.url}"
        )
        if response:
            LOGGER.error(f"Response: {json.dumps(response.json(), indent=2)}")
        return cls(
            f"Snowstorm API returned {response.status_code} status code",
            response,
        )


class PrompterError(Exception):
    """Raised when the prompter encounters an error."""


class PrompterInitError(PrompterError):
    """Raised when prompter can not be initialized"""


## Prompt class
@dataclass(frozen=True, slots=True)
class Prompt:
    """\
Represents a prompt for the LLM agent to answer.

Has option to store API parameters for the answer.
"""

    prompt_message: str | OpenAIMessages
    options: frozenset[SCTDescription] | None = None
    escape_hatch: SCTDescription | None = None
    api_options: frozendict[str, JsonPrimitive] | None = None

    def to_json(self) -> JsonDict:
        """Convert the prompt to a JSON-serializable format for caching."""
        if isinstance(self.prompt_message, str):
            message = self.prompt_message
        else:
            message = [
                {role: text for role, text in message.items()}
                for message in self.prompt_message
            ]

        api_options = dict(self.api_options) if self.api_options else None

        return {
            "prompt_text": json.dumps(message, sort_keys=True),
            "prompt_is_json": not isinstance(self.prompt_message, str),
            "api_options": json.dumps(api_options),
        }  # pyright: ignore[reportReturnType]  # Ruff says it's okay


# Logic classes
## Prompt cache interface
class PromptCache:
    """\
Interface for a prompt cache.

Saves prompts and answers to avoid re-prompting the same questions and wasting
tokens.
"""

    def __init__(self, db_connection: sqlite3.Connection):
        # TODO: form an event queue for this; sqlite does not do well in
        # multi-threaded environments
        self.connection = db_connection
        self.table_name = "prompt"
        self.logger = LOGGER.getChild("PromptCache")

        # Create the table if it does not exist in DB
        table_exists_query = """\
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name=?;
        """
        exists = self.connection.execute(table_exists_query, [self.table_name])
        if not exists.fetchone():
            with open("init_prompt_cache.sql") as f:
                self.connection.executescript(f.read())

    def get(self, model: str, prompt: Prompt) -> str | None:
        """\
Get the answer from the cache for specified model.
"""
        prompt_dict = prompt.to_json()
        api_are_none = prompt_dict["api_options"] is None

        query = f"""
            SELECT response
            FROM {self.table_name}
            WHERE
                model = ? AND
                prompt_text = ? AND
                prompt_is_json = ? AND
                api_options {"IS" if api_are_none else "="} ?
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                query,
                (model, *prompt_dict.values()),
            )
            if answer := cursor.fetchone():
                return answer[0]
            return None
        except (sqlite3.InterfaceError, sqlite3.DatabaseError):
            self.logger.warning(
                "Cache access failed for prompt: "
                + f"{json.dumps(prompt.to_json())}"
            )
            return None

    def remember(self, model: str, prompt: Prompt, response: str) -> None:
        """\
Remember the answer for the prompt for the specified model.
"""
        query = f"""
            INSERT INTO {self.table_name} (
                model,
                prompt_text,
                prompt_is_json,
                api_options,
                response
            )
            VALUES (?, ?, ?, ?, ?)
        """
        # Convert prompt to serializable format
        prompt_dict = prompt.to_json()

        cursor = self.connection.cursor()
        cursor.execute(
            query,
            (
                model,
                *prompt_dict.values(),
                response,
            ),
        )
        self.connection.commit()


## Logic prompt format classes
class PromptFormat(ABC):
    """\
Abstract class for formatting prompts for the LLM agent.
"""

    ROLE = (
        "a domain expert system in clinical terminology who is helping to "
        "build a semantic representation of a concept in a clinical ontology "
        "by providing information about the concept's relationships to other "
        "concepts in the ontology"
    )
    TASK = (
        "to provide information about the given term supertypes, "
        "attributes, attribute values, and other relevant information as "
        "requested, inferring them only from the term meaning and the provided "
        "context"
    )

    REQUIREMENTS = (
        "in addition to providing accurate factually correct information, "
        "it is critically important that you provide answer in a "
        "format that is requested by the system, as answers will "
        "be parsed by a machine. Your answer should ALWAYS end with a line "
        "that says 'The answer is ' and the chosen option. This is the second "
        "time you are being asked the question, as the first time you failed "
        "to adhere to the format. Please make sure to follow the instructions."
    )
    INSTRUCTIONS = (
        "Options that speculate about details not explicitly included in the"
        "term meaning are to be avoided, e.g. term 'operation on abdominal "
        "region' should NOT be assumed to be a laparoscopic operation, as "
        "access method is not specified in the term. It absolutely required to "
        "explain your reasoning when providing answers. The automated system "
        "will look for the last answer surrounded by square brackets, e.g. "
        "[answer], so only one of the options should be selected and returned "
        "in this format. If the question looks like 'What is the topography of "
        "the pulmonary tuberculosis?', and the options are [Lung structure], "
        "[Heart structure], [Kidney structure], the good answer would end with"
        "[Lung structure].' Answers that do not include reasoning are "
        "unacceptable."
    )

    ESCAPE_INSTRUCTIONS = (
        f" If all provided options are incorrect, or imply extra information "
        f"not present explicitly and unambiguously in the term, you must "
        f"explain why each option is incorrect, and finalize the answer with "
        f"the word {EscapeHatch.WORD}. However, if any of the offered terms "
        f"matches the question, you must select it."
    )

    def __init__(self):
        self.logger = LOGGER.getChild("PromptFormat")

    @staticmethod
    def wrap_term(term: str) -> str:
        """Wrap a term in square brackets."""
        return f"[{term}]"

    @abstractmethod
    def form_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to choose the best matching proximal ancestor
for a term.
"""

    @abstractmethod
    def form_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to decide if an attribute is present in a
term.
"""

    @abstractmethod
    def form_attr_value(
        self,
        term: str,
        attribute: SCTDescription,
        options: Iterable[SCTDescription],
        term_context: str | None = None,
        attribute_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        allow_escape: bool = True,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to choose the value of an attribute in a term.
"""

    @abstractmethod
    def form_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to decide if a term is a subtype of an 
existing (primitive) concept.
"""


class VerbosePromptFormat(PromptFormat):
    """\
Default simple verbose prompt format for the LLM agent.

Contains no API options and only string prompts, intended for human prompters.
    """

    def _form_shared_header(self, allow_escape) -> str:
        prompt = ""
        prompt += "You are " + self.ROLE + ".\n\n"
        prompt += "Your assignment is " + self.TASK + ".\n\n"
        prompt += "Please note that " + self.REQUIREMENTS + ".\n\n"
        prompt += "Your exact instructions are:\n\n"
        prompt += self.INSTRUCTIONS + (allow_escape * self.ESCAPE_INSTRUCTIONS)
        prompt += "\n\n"
        return prompt

    def form_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> Prompt:
        prompt = self._form_shared_header(allow_escape)

        prompt += (
            f"Given the term '{term}', what is the closest supertype or exact "
            "equivalent of it's meaning from the following options?"
        )
        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context

        prompt += "\n\nOptions, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            prompt += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                prompt += f": {options_context[option]}"
            prompt += "\n"
        # Remind of the escape hatch, just in case
        if allow_escape:
            prompt += f" - {EscapeHatch.WORD}: " "None of the above\n"
        return Prompt(
            prompt,
            frozenset(options),
            EscapeHatch.WORD if allow_escape else None,
        )

    def form_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> Prompt:
        prompt = self._form_shared_header(allow_escape=False)
        prompt += (
            f"Is the attribute '{attribute}' present in the term '{term}'?"
        )
        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context
        if attribute_context:
            prompt += " Attribute is defined as follows: "
            prompt += attribute_context

        prompt += "\n\n"
        prompt += f"""Options are:
 - {BooleanAnswer.YES}: The attribute is guaranteed present.
 - {BooleanAnswer.NO}: The attribute is absent or not guaranteed present.
"""
        return Prompt(prompt, frozenset((BooleanAnswer.YES, BooleanAnswer.NO)))

    def form_attr_value(
        self,
        term: str,
        attribute: SCTDescription,
        options: Iterable[SCTDescription],
        term_context: str | None = None,
        attribute_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        allow_escape: bool = True,
    ) -> Prompt:
        prompt = self._form_shared_header(allow_escape=allow_escape)
        prompt += (
            f"Choose the value of the attribute '{attribute}' in the term "
            f"'{term}'. The answer should be the best matching option, either "
            f"exact, or a likely supertype of an actual value."
        )
        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context
        if attribute_context:
            prompt += " Attribute is defined as follows: "
            prompt += attribute_context

        prompt += "\n\nOptions, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            prompt += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                prompt += f": {options_context[option]}"
            prompt += "\n"
        # Remind of the escape hatch, just in case
        prompt += f" - {EscapeHatch.WORD}: " "None of the above\n"
        return Prompt(prompt, frozenset(options), EscapeHatch.WORD)

    def form_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> Prompt:
        prompt = self._form_shared_header(allow_escape=False)
        prompt += (
            f"Is the term '{term}' a STRICT SUBTYPE or EXACT MATCH of the "
            f"concept '{prospective_supertype}'? If the term has any "
            f"additional meaning not present in the concept, it can not be "
            f"considered a subtype."
        )

        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context

        if supertype_context:
            prompt += " Following information is provided about the concept: "
            prompt += supertype_context

        prompt += "\n\n"
        prompt += f"""Options are:
 - {BooleanAnswer.YES}: The term is a subtype of the concept.
 - {BooleanAnswer.NO}: The term is not a subtype of the concept.
"""
        return Prompt(prompt, frozenset((BooleanAnswer.YES, BooleanAnswer.NO)))


class OpenAIPromptFormat(PromptFormat):
    """\
Default prompt format for the OpenAI API.

Outputs prompts as JSONs and contains sensible API option defaults.
"""

    default_api_options = frozendict(
        # We want responses to use tokens we provide
        presence_penalty=-0.25,
        # max_tokens=1024,
        timeout=15,
    )
    _UnfinishedPrompt = list[tuple[OpenAIPromptRole, str]]

    def __finalise_prompt(
        self,
        prompt: _UnfinishedPrompt,
        options: frozenset[SCTDescription] | None,
        escape_hatch: SCTDescription | None = None,
    ) -> Prompt:
        history = tuple(
            frozendict({"role": r, "content": t}) for r, t in prompt
        )

        return Prompt(
            history,  # pyright: ignore[reportArgumentType]
            options,
            escape_hatch,
            self.default_api_options,
        )

    def _form_shared_history(self, allow_escape) -> _UnfinishedPrompt:
        prompt = [
            (
                "system",
                "You are " + self.ROLE + ".",
            ),
            (
                "system",
                "Your assignment is " + self.TASK + ".",
            ),
            (
                "system",
                "Please note that " + self.REQUIREMENTS + ".",
            ),
            (
                "system",
                "Your exact instructions are:\n"
                + self.INSTRUCTIONS
                + (allow_escape * self.ESCAPE_INSTRUCTIONS),
            ),
        ]
        return prompt

    def form_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> Prompt:
        if not options:
            self.logger.error("No options provided for supertype prompt")
            self.logger.debug(f"""Prompt context: {term_context}
  - Term: {term}
  - Options: {options}
  - Allowed escape: {allow_escape}
  - Term context: {term_context}
  - Options context: {json.dumps(options_context)}""")
            raise ValueError("No options provided for supertype prompt")

        prompt = self._form_shared_history(allow_escape)
        prompt.append(
            (
                "user",
                (
                    f"Given the term '{term}', what is the closest supertype or "
                    "exact equivalent of it's meaning from the following options?"
                ),
            )
        )
        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        "Following information is provided about the term: "
                        + term_context
                    ),
                )
            )

        options_text = "Options, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            options_text += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                options_text += f": {options_context[option]}"
            options_text += "\n"
        # Remind of the escape hatch, just in case
        if allow_escape:
            options_text += f" - {EscapeHatch.WORD}: " "None of the above\n"

        prompt.append(("user", options_text))

        return self.__finalise_prompt(
            prompt,
            frozenset(options),
            EscapeHatch.WORD if allow_escape else None,
        )

    def form_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> Prompt:
        prompt = self._form_shared_history(allow_escape=False)
        prompt.append(
            (
                "user",
                f"Is the attribute '{attribute}' present in the term '{term}'?",
            )
        )
        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        "Following information is provided about the term: "
                        + term_context
                    ),
                )
            )
        if attribute_context:
            prompt.append(
                (
                    "user",
                    "Attribute is defined as follows: " + attribute_context,
                )
            )

        prompt.append(
            (
                "user",
                f"""Options are:
- {BooleanAnswer.YES}: The attribute is guaranteed present.
- {BooleanAnswer.NO}: The attribute is absent or not guaranteed present.""",
            )
        )

        return self.__finalise_prompt(
            prompt,
            frozenset((BooleanAnswer.YES, BooleanAnswer.NO)),
        )

    def form_attr_value(
        self,
        term: str,
        attribute: SCTDescription,
        options: Iterable[SCTDescription],
        term_context: str | None = None,
        attribute_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        allow_escape: bool = True,
    ) -> Prompt:
        prompt = self._form_shared_history(allow_escape=allow_escape)

        prompt.append(
            (
                "user",
                (
                    f"Choose the value of the attribute '{attribute}' "
                    f"in the term '{term}'. The answer should be the best "
                    f"matching option, either exact, or a likely supertype "
                    f"of an actual value."
                ),
            )
        )

        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        " Following information is provided about the term: "
                        + term_context
                    ),
                )
            )

        if attribute_context:
            prompt.append(
                (
                    "user",
                    (" Attribute is defined as follows: " + attribute_context),
                )
            )

        options_text = "Options, in no particular order:\n"
        for option in sorted(options):  # Sort for cache consistency
            options_text += f" - {self.wrap_term(option)}"
            if options_context and option in options_context:
                options_text += f": {options_context[option]}"
            options_text += "\n"
        # Remind of the escape hatch, just in case
        if allow_escape:
            options_text += f" - {EscapeHatch.WORD}: " "None of the above\n"

        prompt.append(("user", options_text))

        return self.__finalise_prompt(
            prompt,
            frozenset(options),
            EscapeHatch.WORD if allow_escape else None,
        )

    def form_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> Prompt:
        prompt = self._form_shared_history(allow_escape=False)

        prompt.append(
            (
                "user",
                (
                    f"Is the term '{term}' a subtype of the concept "
                    f"'{prospective_supertype}'?"
                ),
            )
        )

        if term_context:
            prompt.append(
                (
                    "user",
                    (
                        " Following information is provided about the term: "
                        + term_context
                    ),
                )
            )

        if supertype_context:
            prompt.append(
                (
                    "user",
                    (
                        " Following information is provided about the concept: "
                        + supertype_context
                    ),
                )
            )

        prompt.append(
            (
                "user",
                f"""Options are:
 - {BooleanAnswer.YES}: The term is a subtype of the concept.
 - {BooleanAnswer.NO}: The term is not a subtype of the concept.""",
            )
        )

        return self.__finalise_prompt(
            prompt,
            frozenset((BooleanAnswer.YES, BooleanAnswer.NO)),
        )


## Logic prompter classes
class Prompter(ABC):
    """\
Interfaces prompts to the LLM agent and parses answers.
"""

    _model_id: str

    def __init__(
        self,
        *args,
        prompt_format: PromptFormat,
        **kwargs,
    ):
        _ = args, kwargs
        self.prompt_format = prompt_format
        self.cache: PromptCache | None = None
        self.logger = LOGGER.getChild(self.__class__.__name__)
        if PARAMS.api.cache_db is not None:
            try:
                conn = sqlite3.connect(
                    PARAMS.api.cache_db, check_same_thread=False
                )
            except Exception as e:
                self.logger.error("Could not connect (create) to the cache DB")
                raise PrompterInitError(e)

            self.cache = PromptCache(conn)

    @staticmethod
    def unwrap_class_answer(
        answer: str,
        options: Iterable[SCTDescription] = (),
        escape_hatch: SCTDescription | None = EscapeHatch.WORD,
    ) -> SCTDescription | EscapeHatch:
        """\
Check if answer has exactly one valid option.

Assumes that the answer is a valid option if it is wrapped in brackets.
"""
        last_line = answer.strip().splitlines()[-1]
        # Try to parse the last line, then the answer as a whole
        look_at = [last_line, answer]

        if not options:
            for text in look_at:
                # Return the answer in brackets, if there is one
                if text.count("[") == text.count("]") == 1:
                    start = text.index("[") + 1
                    end = text.index("]")
                    return SCTDescription(text[start:end])

            raise PrompterError(
                "Could not find a unique option in the answer:", last_line
            )

        wrapped_options = {
            PromptFormat.wrap_term(option): option for option in options
        }

        if escape_hatch is not None:
            wrapped_options = {
                **wrapped_options,
                EscapeHatch.WORD: escape_hatch,
            }

        for text in look_at:
            counts = {}
            for option in wrapped_options:
                counts[option] = text.count(option)

            # Check if there is exactly one option present
            if sum(map(bool, counts.values())) == 1:
                for option, count in counts.items():
                    if count:
                        return (
                            SCTDescription(option[1:-1])
                            if option != escape_hatch
                            else NULL_ANSWER
                        )

            # Return the last encountered option in brackets
            indices: dict[SCTDescription | EscapeHatch, int] = {
                option: text.rfind(wrapped)
                for wrapped, option in wrapped_options.items()
            }
            if any(index != -1 for index in indices.values()):
                return max(indices, key=lambda k: indices.get(k, -1))

        raise PrompterError(
            "Could not find a unique option in the answer:", last_line
        )

    @staticmethod
    def unwrap_bool_answer(
        answer: str,
        yes: str = BooleanAnswer.YES,
        no: str = BooleanAnswer.NO,
    ) -> bool:
        """\
Check if the answer contains a yes or no option.
"""
        last_line = answer.strip().splitlines()[-1]
        if yes in last_line and no not in last_line:
            return True
        elif no in last_line and yes not in last_line:
            return False
        else:
            raise PrompterError(
                "Could not find an unambiguous boolean answer in the response"
            )

    async def prompt_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> SCTDescription | EscapeHatch:
        """\
Prompt the model to choose the best matching proximal ancestor for a term.
"""
        # Construct the prompt
        prompt: Prompt = self.prompt_format.form_supertype(
            term, options, allow_escape, term_context, options_context
        )
        self.logger.debug(f"Constructed prompt: f{prompt.prompt_message}")

        if cached_answer := self.cache_get(prompt):
            answer = self.unwrap_class_answer(
                cached_answer,
                options,
                EscapeHatch.WORD if allow_escape else None,
            )
        else:
            # Get the answer
            answer = await self._prompt_class_answer(
                allow_escape, options, prompt
            )
        self.logger.info(f"Agent answer: {answer} is a supertype of {term}")
        return answer

    async def prompt_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> bool:
        prompt: Prompt = self.prompt_format.form_attr_presence(
            term, attribute, term_context, attribute_context
        )

        if cached_answer := self.cache_get(prompt):
            answer = self.unwrap_bool_answer(cached_answer)
        else:
            answer = await self._prompt_bool_answer(prompt)
        self.logger.info(
            f"Agent answer: The attribute '{attribute}' is "
            f"{'present' if answer else 'absent'} in '{term}'"
        )
        return answer

    async def prompt_attr_value(
        self,
        term: str,
        attribute: SCTDescription,
        options: Iterable[SCTDescription],
        term_context: str | None = None,
        attribute_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        allow_escape: bool = True,
    ) -> SCTDescription | EscapeHatch:
        prompt: Prompt = self.prompt_format.form_attr_value(
            term,
            attribute,
            options,
            term_context,
            attribute_context,
            options_context,
            allow_escape,
        )
        """\
Prompt the model to choose the value of an attribute in a term.
"""

        if cached_answer := self.cache_get(prompt):
            answer = self.unwrap_class_answer(
                cached_answer,
                options,
                EscapeHatch.WORD if allow_escape else None,
            )
        else:
            answer = await self._prompt_class_answer(
                allow_escape, options, prompt
            )

        self.logger.info(
            f"Agent answer: The value of the attribute '{attribute}' in "
            f"'{term}' is '{answer}'"
        )
        return answer

    async def prompt_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> bool:
        """\
Prompt the model to decide if a term is a subtype of a prospective supertype.

Only meant to be used for Primitive concepts: use Bouzyges.check_subsumption for
Fully Defined concepts.
"""
        prompt: Prompt = self.prompt_format.form_subsumption(
            term, prospective_supertype, term_context, supertype_context
        )

        if cached_answer := self.cache_get(prompt):
            answer = self.unwrap_bool_answer(cached_answer)
        else:
            answer = await self._prompt_bool_answer(prompt)

        self.logger.info(
            f"From cache: The term '{term}' is "
            f"{'a subtype' if answer else 'not a subtype'} "
            f"of '{prospective_supertype}'"
        )
        return answer

    def cache_remember(self, prompt: Prompt, answer: str) -> None:
        if self.cache:
            self.cache.remember(self._model_id, prompt, answer)

    def cache_get(self, prompt: Prompt) -> str | None:
        if self.cache:
            return self.cache.get(self._model_id, prompt)
        return None

    # Following methods are abstract and represent common queries to the model
    @abstractmethod
    async def _prompt_bool_answer(self, prompt: Prompt) -> bool:
        """\
Send a prompt to the counterpart agent to obtain the answer
"""

    @abstractmethod
    async def _prompt_class_answer(
        self,
        allow_escape: bool,
        options: Iterable[SCTDescription],
        prompt: Prompt,
    ) -> SCTDescription | EscapeHatch:
        """\
Send a prompt to the counterpart agent to obtain a single choice answer.
"""

    @abstractmethod
    def ping(self) -> bool:
        """\
Check if the API is available.
"""

    @abstractmethod
    def report_usage(self) -> None:
        """\
Report the usage of the API to the provider.

Form is not specified, as it is provider-specific.
"""


class HumanPrompter(Prompter):
    """\
A test prompter that interacts with a human to get answers.
"""

    _model_id = "human"

    def __init__(self, *args, prompt_function: Callable[[str], str], **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_function = prompt_function

    async def _prompt_class_answer(self, allow_escape, options, prompt):
        while True:
            brain_answer = self.prompt_function("Answer: ").strip()
            try:
                answer = self.unwrap_class_answer(
                    brain_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
                self.cache_remember(prompt, brain_answer)
                return answer

            except PrompterError as e:
                logging.error("Error: %s", e)

    async def _prompt_bool_answer(self, prompt: Prompt) -> bool:
        while True:
            brain_answer = self.prompt_function("Answer: ").strip()
            try:
                answer = self.unwrap_bool_answer(brain_answer)
                self.cache_remember(prompt, brain_answer)
                return answer
            except PrompterError as e:
                logging.error("Error: %s", e)

    def ping(self) -> bool:
        self.prompt_function("Press Enter to confirm you are here")
        return True

    def report_usage(self) -> None:
        self.logger.info(
            "No usage to report, as this is a human prompter. Stay "
            "hydrated and have a good day!"
        )


class OpenAIPrompter(Prompter):
    """\
A prompter that interfaces with the OpenAI API using.
"""

    def __init__(
        self,
        *args,
        model: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._model_id = model
        self._init_client(*args, **kwargs)

        self._estimated_token_usage: dict[Prompt, int] = {}
        self._actual_token_usage: dict[Prompt, int] = {}
        if tiktoken is not None:
            try:
                self._estimation_encoding = tiktoken.encoding_for_model(
                    self._model_id
                )
            except KeyError:
                self.logger.warning("Model not found in the token")
                self._estimation_encoding = None
        else:
            self.logger.warning(
                "Tiktoken not installed, can't track token usage"
            )
            self._estimation_encoding = None

    def _init_client(self, *args, **kwargs):
        self.logger.info("Initializing the OpenAI API client...")
        _ = args, kwargs
        # API Key will be picked up from env variables
        # self._api_key = api_key
        if not os.getenv("OPENAI_API_KEY"):
            raise PrompterInitError(
                f"Can not initialize {self} without OPENAI_API_KEY"
            )

        self._client = openai.AsyncOpenAI()
        self._ping_headers = {}

    async def _ping(self):
        try:
            return await self._client.models.list(
                extra_headers=self._ping_headers, timeout=5
            )
        except Exception as e:
            self.logger.warning(f"Connection timed out: {e}")
            raise

    def ping(self) -> bool:
        self.logger.info("Pinging the OpenAI API...")
        # Ping by retrieving the list of models
        try:
            models = asyncio.run(self._ping())
        except Exception as e:
            self.logger.warning(f"API is not available: {e}")
            return False

        response: dict = models.model_dump()
        success = response.get("data", []) != []
        if success:
            self.logger.info("API is available")
            self.logger.debug(
                f"Models: {json.dumps(response['data'], indent=2)}"
            )
            if not any(self._model_id == obj["id"] for obj in response["data"]):
                self.logger.warning(
                    f"'{self._model_id}' is not present in the API response!"
                )
                return False
            return True

        self.logger.warning("API is not available")
        return False

    async def _prompt_bool_answer(self, prompt: Prompt) -> bool:
        return await self._prompt_answer(prompt, self.unwrap_bool_answer)

    async def _prompt_class_answer(self, allow_escape, options, prompt):
        return await self._prompt_answer(
            prompt,
            lambda x: self.unwrap_class_answer(
                x, options, EscapeHatch.WORD if allow_escape else None
            ),
        )

    async def _prompt_answer(
        self, prompt: Prompt, parser: Callable[[str], T], parse_retries_left=3
    ) -> T:
        self.logger.info("Trying cache for answer...")
        if cached_answer := self.cache_get(prompt):
            answer = parser(cached_answer)
            self.logger.info("Cache hit!")
            return answer
        else:
            self.logger.info("Cache miss")

        self.logger.info("Prompting the OpenAI API for an answer...")

        if self._estimation_encoding:
            token_count = len(
                self._estimation_encoding.encode(
                    json.dumps(prompt.prompt_message)
                )
            )
            self._estimated_token_usage[prompt] = token_count
            self.logger.debug(
                f"Estimated token usage for prompt: {token_count}"
            )
        else:
            self.logger.warning(
                "Token usage will not be estimated: unknown model"
            )

        self.logger.debug(
            f"Prompt message {json.dumps(prompt.prompt_message, indent=2)}"
        )

        if isinstance(prompt.prompt_message, str):
            messages = [
                {
                    "role": "system",
                    "message": prompt.prompt_message,
                },
            ]
        else:
            messages = prompt.prompt_message

        try:
            brain_answer = await self._get_completion(
                messages=messages,  # type: ignore
                **(prompt.api_options or {}),
            )
        except openai.APIError as e:
            self.logger.error(f"API error: {e}")
            raise PrompterError("Failed to get a response from the API")

        response_message = brain_answer.choices[0].message.content
        if self._estimation_encoding:
            token_count = len(
                self._estimation_encoding.encode(response_message)
            )
            self._estimated_token_usage[prompt] = (
                self._estimated_token_usage.get(prompt, 0) + token_count
            )
            self.logger.debug(
                f"Estimated token usage for answer: {token_count}"
            )

        self._actual_token_usage[prompt] = (
            self._actual_token_usage.get(prompt, 0)
            + brain_answer.usage.total_tokens
        )

        self.logger.debug(f"Literal response: {response_message}")
        try:
            answer = parser(response_message)
            self.cache_remember(prompt, response_message)
            return answer
        except PrompterError as e:
            # Recursively call self if LLM fails to provide a parsable answer
            self.logger.error(f"Error parsing response: {e}")
            if parse_retries_left > 0:
                self.logger.warning(
                    f"Retrying parsing the answer, "
                    f"attempts left: {parse_retries_left}"
                )
                return await self._prompt_answer(
                    prompt, parser, parse_retries_left - 1
                )
            raise PrompterError("Failed to parse the answer")

    @tenacity.retry(wait=tenacity.wait_random_exponential(multiplier=1, max=60))
    async def _get_completion(self, messages, **kwargs):
        return await self._client.chat.completions.create(
            messages=messages,  # type: ignore
            model=self._model_id,
            **kwargs,
        )

    def report_usage(self) -> None:
        self.logger.info("Reporting usage to the OpenAI API...")
        if self._estimated_token_usage:
            n_prompts = len(self._estimated_token_usage)
            total_tokens = sum(self._estimated_token_usage.values())
            self.logger.info(
                f"Estimation: reporting {n_prompts} prompts with a total of "
                f"{total_tokens} tokens"
            )
        else:
            self.logger.warning("No estimation of token usage to report")

        n_prompts = len(self._actual_token_usage)
        total_tokens = sum(self._actual_token_usage.values())
        self.logger.info(
            f"Actual: reporting {n_prompts} prompts with a total of "
            f"{total_tokens} tokens"
        )


class OpenAIAzurePrompter(OpenAIPrompter):
    """\
A prompter that interfaces with the OpenAI API using Azure.
"""

    DEFAULT_VERSION = "2024-06-01"

    def _init_client(self, api_key: str, azure_endpoint: str):
        self.logger.info("Initializing the Azure API client...")
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=self.DEFAULT_VERSION,
        )
        self._ping_headers = {"api-key": self._client.api_key}


class SnowstormAPI:
    TARGET_CODESYSTEM = "SNOMEDCT"
    CONTENT_TYPE_PREFERENCE = "NEW_PRECOORDINATED", "PRECOORDINATED", "ALL"
    PAGINATION_STEP = 100
    MAX_BAD_PARENT_QUERY = 32

    def __init__(self, url: Url):
        # Debug
        self.__start_time = datetime.datetime.now()
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.url: Url = url
        self.mrcm_entries: list[MRCMDomainRefsetEntry] = []

        # Cache repetitive queries
        self.__concepts_cache: dict[SCTID, Concept] = {}
        self.__subsumptions_cache: dict[tuple[SCTID, SCTID], bool] = {}

    @classmethod
    async def init(cls, url: Url) -> SnowstormAPI:
        snowstorm = cls(url)
        await snowstorm.ping()

        # Load MRCM entries
        snowstorm.logger.info("Loading MRCM Domain Reference Set entries")
        domain_entries = await snowstorm.get_mrcm_domain_reference_set_entries()
        snowstorm.logger.info(f"Total entries: {len(domain_entries)}")
        snowstorm.mrcm_entries = [
            MRCMDomainRefsetEntry.from_json(entry)
            for entry in domain_entries
            if SCTID(entry["referencedComponent"]["conceptId"])
            in WHITELISTED_SUPERTYPES
        ]

        entries_msg = ["MRCM entries:"]
        for entry in snowstorm.mrcm_entries:
            entries_msg.append(" - " + entry.term + ":")
            entries_msg.append("    - " + entry.domain_constraint)
            entries_msg.append("    - " + entry.guide_link)
        snowstorm.logger.debug("\n".join(entries_msg))

        return snowstorm

    async def ping(self) -> bool:
        # Get the main branch path (and try connecting)
        try:
            self.logger.info("Snowstorm Version: " + await self.get_version())
            self.branch_path: BranchPath = await self.get_main_branch_path()
        except Exception as e:
            self.logger.error(f"Could not get branch from Snowstorm API: {e}")
            raise
        # Log the version
        self.logger.info("Using branch path: " + self.branch_path)
        return True

    async def _get(self, *args, **kwargs) -> requests.Response:
        """\
Wrapper for requests.get that prepends known url and
raises an exception on non-200 responses.
"""
        # Check for the timeout
        if PARAMS.prof.stop_profiling_after_seconds:
            elapsed = datetime.datetime.now() - self.__start_time
            seconds = elapsed.total_seconds()
            if seconds > PARAMS.prof.stop_profiling_after_seconds:
                raise ProfileMark("Time limit exceeded")

        # Include the known url
        if "url" not in kwargs:
            args = (self.url + args[0], *args[1:])
        else:
            kwargs["url"] = self.url + kwargs["url"]

        kwargs["headers"] = kwargs.get("headers", {})
        kwargs["headers"]["Accept"] = "application/json"

        try:
            response = requests.get(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Could not connect to Snowstorm API: {e}")
            raise BouzygesError(f"Could not connect to Snowstorm API: {e}")

        if not response.ok:
            self.logger.error(
                f"Request failed: {response.status_code} {response.reason} "
                f"for {response.url}"
            )
            self.logger.error(
                f"Response: {json.dumps(response.json(), indent=2)}"
            )
            raise SnowstormRequestError.from_response(response)

        return response

    async def _get_collect(self, *args, **kwargs) -> list:
        """\
Wrapper for requests.get that collects all items from a paginated response.
"""
        # TODO: request multiple pages in parallel if possible
        total = None
        offset = 0
        step = self.PAGINATION_STEP
        collected_items = []

        while total is None or offset < total:
            kwargs["params"] = kwargs.get("params", {})
            kwargs["params"]["offset"] = offset
            kwargs["params"]["limit"] = step

            response = await self._get(*args, **kwargs)

            collected_items.extend(response.json()["items"])
            total = response.json()["total"]
            offset += step

        return collected_items

    async def get_version(self) -> str:
        response = await self._get("version")
        return response.json()["version"]

    async def get_main_branch_path(self) -> BranchPath:
        # Get codesystems and look for a target
        response = await self._get("codesystems")

        for codesystem in response.json()["items"]:
            if codesystem["shortName"] == self.TARGET_CODESYSTEM:
                # TODO: double-check by module contents
                return BranchPath(codesystem["branchPath"])

        raise SnowstormAPIError(
            f"Target codesystem {self.TARGET_CODESYSTEM} is not present"
        )

    async def get_concept(self, sctid: SCTID) -> Concept:
        """\
Get full concept information.
"""
        if sctid in self.__concepts_cache:
            return self.__concepts_cache[sctid]
        response = await self._get(
            url=f"browser/{self.branch_path}/concepts/{sctid}",
            params={"activeFilter": True},
        )
        concept = Concept.from_json(response.json())
        self.__concepts_cache[sctid] = concept
        return concept

    async def get_branch_info(self) -> dict:
        response = await self._get(f"branches/{self.branch_path}")
        return response.json()

    async def get_attribute_suggestions(
        self, parent_ids: Iterable[SCTID]
    ) -> dict[SCTID, AttributeConstraints]:
        response = await self._get(
            url="mrcm/" + self.branch_path + "/domain-attributes",
            params={
                "parentIds": [*parent_ids],
                "proximalPrimitiveModeling": True,  # Maybe?
                # Filter post-coordination for now
                "contentType": "ALL",
            },
        )

        return {
            SCTID(attr["id"]): AttributeConstraints.from_json(attr)
            for attr in response.json()["items"]
            if SCTID(attr["id"]) != IS_A  # Exclude hierarchy
        }

    async def get_mrcm_domain_reference_set_entries(
        self,
    ) -> list[dict]:
        collected_items = await self._get_collect(
            url=f"{self.branch_path}/members",
            params={
                "referenceSet": MRCM_DOMAIN_REFERENCE_SET_ECL,
                "active": True,
            },
        )
        return collected_items

    def _range_constraint_to_parents(
        self,
        rc: ECLExpression,
    ) -> dict[SCTID, SCTDescription]:
        """\
This is an extremely naive implementation that assumes that the range constraint
is always a disjunction of parent SCTIDs

As I am not expecting to have to parse ECL anywhere else now, this will have to
do
"""
        # TODO: Hook up to the ECL parser

        # Check the assumption; if it fails, raise an error
        SUBSUMPTION = "<< "
        SCTID_ = r"(?P<sctid>\d{6,}) "  # Intentional capture group
        TERM_ = r"\|(?P<term>.+?) \([a-z]+(?: (?:[a-z]+|\/))*\)\|"
        subsumption_constraint = re.compile(SUBSUMPTION + SCTID_ + TERM_)

        parents: dict[SCTID, SCTDescription] = {}
        failure = False
        for part in rc.split(" OR "):
            if not subsumption_constraint.fullmatch(part):
                failure = True

            if matched := subsumption_constraint.match(part):
                parents[SCTID(matched.group("sctid"))] = SCTDescription(
                    matched.group("term")
                )
            else:
                failure = True

            if failure:
                self.logger.error(
                    f"{rc} is not a simple disjunction of SCTIDs!"
                )
                raise NotImplementedError

        # If there is more than one available parent, remove the root
        # Only seen in post-coordination ranges so far, but just in case
        if len(parents) > 1:
            parents.pop(ROOT_CONCEPT, None)

        return parents

    def get_attribute_values(
        self, portrait: SemanticPortrait, attribute: SCTID
    ) -> dict[SCTID, SCTDescription]:
        # First, obtain the range constraints
        ranges = portrait.relevant_constraints[attribute].attribute_range

        # Choose preferred content type
        for ctype in self.CONTENT_TYPE_PREFERENCE:
            for r in ranges:
                if r.contentType == ctype:
                    return self._range_constraint_to_parents(r.range_constraint)

        # No range of allowed content types found!
        self.logger.warning(f"No range constraint found for {attribute}")
        return {}

    async def get_concept_children(
        self,
        parent: SCTID,
        require_property: Mapping[str, JsonPrimitive] | None = None,
    ) -> dict[SCTID, SCTDescription]:
        response = await self._get(
            url=f"browser/{self.branch_path}" + f"/concepts/{parent}/children",
        )

        children: dict[SCTID, SCTDescription] = {}
        for child in response.json():
            skip: bool = (require_property is not None) and not all(
                child.get(k) == v for k, v in require_property.items()
            )
            if not skip:
                id = SCTID(child["conceptId"])
                term = SCTDescription(child["pt"]["term"])
                children[id] = term

        return children

    async def is_concept_descendant_of(
        self,
        child: SCTID,
        parent: SCTID,
        self_is_parent: bool = True,
    ) -> bool:
        """\
Implements a subsumption check for concepts in the SNOMED CT hierarchy.
Returns True if the child is a descendant of the parent, and False otherwise.

See: https://confluence.ihtsdotools.org/display/DOCTSG/4.5+Get+and+Test+\
Concept+Subtypes+and+Supertypes
"""
        if child == parent:
            return self_is_parent

        if (child, parent) in self.__subsumptions_cache:
            return self.__subsumptions_cache[(child, parent)]

        response = await self._get(
            url=f"{self.branch_path}/concepts/",
            params={
                "ecl": f"<{parent}",  # Is a subtype of
                "conceptIds": [child],  # limit to known child
            },
        )

        out = bool(response.json()["total"])  # Should be 1 or 0

        self.__subsumptions_cache[(child, parent)] = out  # Cache the result
        return out

    async def filter_bad_descendants(
        self, children: Iterable[SCTID], bad_parents: Iterable[SCTID]
    ) -> set[SCTID]:
        """\
Filter out children that are descendants of bad parents and return the rest.
"""
        if not bad_parents:
            return set(children)

        # First, remove the direct matches and cached hits
        out = set(children) - set(bad_parents)
        known_bad = set()
        for child, bad_parent in itertools.product(out, bad_parents):
            if self.__subsumptions_cache.get((child, bad_parent)):
                known_bad.add(child)
        out -= known_bad

        if not out:
            return out

        # Batch request, because Snowstorm hates long urls
        # TODO: Parallelize?
        for bad_batch in itertools.batched(
            bad_parents, self.MAX_BAD_PARENT_QUERY
        ):
            expression = " OR ".join(f"<{b_p}" for b_p in sorted(bad_batch))
            actual_children = await self._get_collect(
                url=f"{self.branch_path}/concepts/",
                params={
                    "activeFilter": True,
                    "ecl": expression,
                    "returnIdOnly": True,
                    "conceptIds": sorted(out),  # limit to known children
                },
            )
            out -= set(SCTID(c) for c in actual_children)
            if not out:
                break

        return out

    async def is_attr_val_descendant_of(
        self, child: AttributeRelationship, parent: AttributeRelationship
    ) -> bool:
        """\
Check if the child attribute-value pair is a subtype of the parent.
"""
        ANSWER = [False, False]

        async def check_attr():
            ANSWER[0] = await self.is_concept_descendant_of(
                child.attribute, parent.attribute
            )

        async def check_val():
            ANSWER[1] = await self.is_concept_descendant_of(
                child.value, parent.value
            )

        asyncio.gather(check_attr(), check_val())
        return all(ANSWER)

    async def remove_redundant_ancestors(
        self, portrait: SemanticPortrait
    ) -> None:
        """\
Remove ancestors that are descendants of other ancestors.
"""
        redundant_ancestors = set()
        ancestor_matrix = itertools.combinations(portrait.ancestor_anchors, 2)

        async def check_pair(pair):
            ancestor, other = pair
            if await self.is_concept_descendant_of(other, ancestor):
                redundant_ancestors.add(ancestor)

        asyncio.gather(*map(check_pair, ancestor_matrix))

        portrait.ancestor_anchors -= redundant_ancestors

    async def get_concept_ppp(self, concept: SCTID) -> set[SCTID]:
        """\
Get a concept's Proximal Primitive Parents
"""
        response = await self._get(
            f"{self.branch_path}/concepts/{concept}/normal-form",
        )

        focus_concepts_string = response.json()["expression"].split(" : ")[0]
        self.logger.debug(f"PPP for {concept}: {focus_concepts_string}")
        concepts = map(SCTID, focus_concepts_string.split(" + "))

        out = set()

        async def process_concept(concept):
            nonlocal out
            concept_info = await self.get_concept(concept)
            if concept_info.defined:
                out |= await self.get_concept_ppp(concept)
            else:
                out.add(concept)

        asyncio.gather(*map(process_concept, concepts))
        return out

    async def check_inferred_subsumption(
        self, parent_predicate: Concept, portrait: SemanticPortrait
    ) -> bool:
        """\
Check if the particular portrait can be a subtype of a parent concept.

Note that subsumption is checked for concepts regardless of definition status;
Primitive concepts will report subsumption as True, but it needs to be confirmed
manually/with LLM.
"""
        self.logger.debug(
            "Checking subsumption for "
            + portrait.source_term
            + " under "
            + parent_predicate.pt
        )

        # To be considered eligible as a descendant, all the predicate's PPP
        # must be ancestors of at least one anchor
        unmatched_predicate_ppp: set[SCTID] = await self.get_concept_ppp(
            parent_predicate.sctid
        )

        async def check_anchor(anchor):
            nonlocal unmatched_predicate_ppp
            if not unmatched_predicate_ppp:
                return
            anchor_matched_ppp = set()

            async def check_ppp(ppp):
                if await self.is_concept_descendant_of(anchor, ppp):
                    anchor_matched_ppp.add(ppp)

            asyncio.gather(*map(check_ppp, unmatched_predicate_ppp))
            unmatched_predicate_ppp -= anchor_matched_ppp

        asyncio.gather(*map(check_anchor, portrait.ancestor_anchors))

        if unmatched_predicate_ppp:
            self.logger.debug(
                f"Does not satisfy {len(unmatched_predicate_ppp)} "
                f"PPP constraints"
            )
            return False

        # For now, we do not worry about the groups; we may have to once
        # we allow multiple of a same attribute
        unmatched_concept_relationships: set[AttributeRelationship] = set()
        for group in parent_predicate.groups:
            unmatched_concept_relationships |= group.relationships
        unmatched_concept_relationships |= parent_predicate.ungrouped

        # TODO: asyncify
        for av in portrait.attributes.items():
            p_rel = AttributeRelationship(*av)
            matched_attr: set[AttributeRelationship] = set()
            for c_rel in unmatched_concept_relationships:
                if await self.is_attr_val_descendant_of(p_rel, c_rel):
                    matched_attr.add(c_rel)
            unmatched_concept_relationships -= matched_attr
            if not unmatched_concept_relationships:
                # Escape early if all relationships are matched
                break

        if unmatched := len(unmatched_concept_relationships):
            self.logger.debug(
                f"Does not satisfy {unmatched} attribute constraints"
            )
            return False

        self.logger.debug("All constraints are satisfied")
        if not parent_predicate.defined:
            self.logger.debug(
                "Concept is primitive: subsumption must be confirmed manually"
            )
        return True


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

    def __init__(self, path: str) -> None:
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.path = path
        self.content: pd.DataFrame | None = None

    def read(self) -> None:
        """\
Read the file
"""
        decode_sep = {
            "Tab": "\t",
        }
        encoded_sep = PARAMS.read.sep

        try:
            df = pd.read_csv(
                self.path,
                dtype=str,
                sep=decode_sep.get(encoded_sep, encoded_sep),
                quotechar=PARAMS.read.quotechar,
            )
        except Exception as e:
            self.logger.error(f"Could not read {self.path}: {e}")
            raise

        self.logger.info(f"Read {len(df)} rows from {self.path}")
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


class TestFileReader(unittest.TestCase):
    def setUp(self) -> None:
        self.row = pd.Series(
            {
                "vocab": "ICD-42",
                "code": 123456,
                "term": "Test term",
                "foo": "Test context",
                "bar": "Another context",
            }
        )
        return super().setUp()

    def test_parse(self):
        portrait = FileReader._portrait_from_row(self.row)
        self.assertEqual(portrait.source_term, "Test term")
        self.assertIsNot(portrait.context, None)
        self.assertEqual(
            set(portrait.context),  # type: ignore
            {"Test context", "Another context"},
        )
        self.assertEqual(
            portrait.metadata, frozendict(vocab="ICD-42", code="123456")
        )


class FileWriter:
    """\
A class to write resulting files.
"""

    def __init__(self, path: str, format: OutFormat = "SCG") -> None:
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.path = path
        self.content: pd.DataFrame | Json | None = None

        self.write_chosen: Callable[[Iterable[WrappedResult]], None]
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
                sep=PARAMS.write.sep,
                quotechar=PARAMS.write.quotechar,
                quoting=PARAMS.write.quoting,
                na_rep="",
            )
        except Exception as e:
            self.logger.error(f"Could not write to {self.path}: {e}")
            raise

        self.logger.info(f"Written to {self.path}")

    def to_snomed_compositional_grammar(
        self, results: Iterable[WrappedResult]
    ) -> None:
        """\
Write the results of term evaluation as a table of SNOMED CT Post-Coordinated
Expressions.

Does not normalize nor verify the expressions, nor considers grouping rules;
some tools like CSIRO Ontoserver can do that. Unfortunately, normalization rules
are not formally defined.
"""
        self.logger.debug("Writing to SCG format")
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

    def to_concept_relationship_stage(
        self, results: Iterable[WrappedResult]
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

    def to_json(self, results: Iterable[WrappedResult]) -> None:
        """\
Write the results of term evaluation as a JSON file.

JSON schema:
    {
        "items": [
            {
                "term": "Pyogenic abscess of liver",
                "attributes": [
                    {
                        "attribute": 363698007,
                        "value": 10200004
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
        for result in results:
            portrait = result.portrait
            map = result.name_map
            row = {
                "term": portrait.source_term,
                "attributes": [
                    {"attribute": k, "value": v}
                    for k, v in portrait.attributes.items()
                ],
                "proximal_ancestors": [
                    {"conceptId": k, "pt": map.get(k, "Unknown")}
                    for k in portrait.ancestor_anchors
                ],
                "scg": portrait.to_scg(),
                "metadata": dict(portrait.metadata),
            }
            dicts.append(row)
        self.content = {"items": dicts}
        self.logger.debug("Content ready")

        with open(self.path, "w") as f:
            try:
                json.dump(self.content, f, indent=2)
            except Exception as e:
                self.logger.error(f"Could not write to {self.path}: {e}")
                raise

        self.logger.info(f"Written to {self.path}")


# Main logic host
class Bouzyges:
    """\
Main logic host for the Bouzyges system.
"""

    def __init__(
        self,
        snowstorm: SnowstormAPI,
        prompter: Prompter,
        portraits: Iterable[SemanticPortrait],
    ):
        self.snowstorm = snowstorm
        self.prompter = prompter
        self.portraits = {p.source_term: p for p in portraits}

        self.logger = LOGGER.getChild(self.__class__.__name__)

        self.results: list[WrappedResult] = []

    @staticmethod
    async def read_file(logger, prep_dict, ready_callback) -> None:
        """\
Read the input file and parse it into a list of SemanticPortrait objects.
"""
        # Read file for portraits
        if not PARAMS.read.file:
            raise BouzygesError("No input file specified!")
        logger.info("Reading input file...")
        reader = FileReader(PARAMS.read.file)
        reader.read()
        if reader.content is None:
            raise BouzygesError("Could not read the input file")
        portraits = reader.parse()
        logger.info(f"Read {len(portraits)} portraits")
        prep_dict["portraits"] = portraits
        ready_callback()

    @staticmethod
    async def get_snowstorm(logger, prep_dict, ready_callback):
        """\
Initialize the SnowstormAPI object.
"""
        logger.info("Initializing Snowstorm API...")
        try:
            snowstorm = await SnowstormAPI.init(PARAMS.api.snowstorm_url)
        except Exception as e:
            logger.error("Could not connect to Snowstorm API:", e)
            raise
        prep_dict["snowstorm"] = snowstorm
        ready_callback()

    @staticmethod
    async def get_prompter(logger, prep_dict, ready_callback):
        logger.info("Initializing prompter...")
        prompter: Prompter
        match PARAMS.api.prompter:
            case "openai":
                prompter = OpenAIPrompter(
                    prompt_format=OpenAIPromptFormat(),
                    model=PARAMS.api.llm_model_id,
                )

            case "azure":
                prompter = OpenAIAzurePrompter(
                    prompt_format=OpenAIPromptFormat(),
                    api_key=PARAMS.env.AZURE_API_KEY,
                    azure_endpoint=PARAMS.env.AZURE_API_ENDPOINT,
                    model=PARAMS.api.llm_model_id,
                )
            case "human":
                prompter = HumanPrompter(
                    prompt_function=input,
                    prompt_format=VerbosePromptFormat(),
                )

            case _:
                raise ValueError("Invalid prompter option")
        prep_dict["prompter"] = prompter
        logger.info("Prompter initialized")
        ready_callback()

    @classmethod
    async def prepare(cls, progress_callback) -> Self:
        logger = LOGGER.getChild(cls.__name__)
        progress_callback(0, 3)

        prep_dict = {}

        def report_completion():
            progress_callback(len(prep_dict), 3)

        snowstorm_task = cls.get_snowstorm(logger, prep_dict, report_completion)
        prompter_task = cls.get_prompter(logger, prep_dict, report_completion)
        read_file_task = cls.read_file(logger, prep_dict, report_completion)

        await asyncio.gather(snowstorm_task, prompter_task, read_file_task)

        return cls(**prep_dict)

    async def _run(self, progress_callback):
        """Main routine"""
        start_time = datetime.datetime.now()
        self.logger.info(f"Started at: {start_time}")

        workers = [
            _BouzygesWorker(i, portrait, self)
            for i, portrait in enumerate(self.portraits.values())
        ]

        done: list[_BouzygesWorker] = []

        def report_progress(worker: _BouzygesWorker):
            done.append(worker)
            progress_callback(len(done), len(workers))

        self.results = await asyncio.gather(
            *map(lambda w: w.run(report_progress), workers)
        )

        self.logger.info("Routine finished")
        self.logger.info(
            f"Time taken (s): "
            f"{(datetime.datetime.now() - start_time).total_seconds()}"
        )

    async def output(self, progress_callback) -> None:
        """\
Output the results of the Bouzyges system.
"""
        progress_callback(0, 1)
        if PARAMS.write.file is None:
            raise BouzygesError("No output file specified")

        out_path = os.path.join(PARAMS.out_dir, PARAMS.write.file)
        writer = FileWriter(out_path, format=PARAMS.format)
        writer.write_chosen(self.results)
        progress_callback(1, 1)

    async def run(self, progress_callback):
        """\
Run the Bouzyges system.
"""
        if PARAMS.prof:
            with cProfile.Profile() as prof:
                try:
                    await self._run(progress_callback)
                except ProfileMark:
                    pass
                finally:
                    stats = pstats.Stats(prof)
                    stats.sort_stats(pstats.SortKey.TIME)
                    stats.dump_stats("stats.prof")
        else:
            await self._run(progress_callback)


class _BouzygesWorker:
    """\
Worker thread for the Bouzyges system, performing the main logic. on a single
source term.
"""

    def __init__(
        self, idx: int, portrait: SemanticPortrait, bouzyges: Bouzyges
    ):
        self.source_term = portrait.source_term
        self.portrait = portrait

        term_abbrev = "".join(
            map(
                lambda w: re.sub(r"\W", "", w)[0].upper(),
                self.source_term.split(),
            )
        )
        self.logger = bouzyges.logger.getChild(f"Worker {idx}:{term_abbrev}")
        self.snowstorm = bouzyges.snowstorm
        self.prompter = bouzyges.prompter
        self.__mrcm_entries = bouzyges.snowstorm.mrcm_entries

        self.logger.info(f"Worker for '{self.source_term}' is ready")

    async def run(self, report_progress) -> WrappedResult:
        """\
Run the worker thread.
"""
        start_time = datetime.datetime.now()
        self.logger.info(f"Started at: {start_time}")
        try:
            return await self._run(report_progress)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            raise
        finally:
            self.logger.info(
                f"Time taken (s): "
                f"{(datetime.datetime.now() - start_time).total_seconds()}"
            )
            self.prompter.report_usage()

    async def _run(self, report_progress) -> WrappedResult:
        await self.initialize_supertypes()
        await self.populate_attribute_candidates()
        await self.populate_unchecked_attributes()
        await self.update_existing_attr_values()

        changes_made = updated = True
        while changes_made:
            cycles = 0
            while updated:
                updated = await self.update_anchors()
                cycles += updated
            changes_made = bool(cycles)

        await self.snowstorm.remove_redundant_ancestors(self.portrait)

        # Log resulting supertypes
        attr_message = []
        attr_message += [f"'{self.source_term}' Attributes:"]
        for attribute, value in self.portrait.attributes.items():
            attr_message.append(f" - {attribute}={value}")
        self.logger.info("\n".join(attr_message))

        supr_message = []
        supr_message += [f"'{self.source_term}' Supertypes:"]
        anchors: dict[SCTID, SCTDescription] = {}

        async def get_anchor_info(anchor):
            concept = await self.snowstorm.get_concept(anchor)
            supr_message.append(f" - {concept.sctid} {concept.pt}")
            anchors[concept.sctid] = concept.pt

        asyncio.gather(*map(get_anchor_info, self.portrait.ancestor_anchors))
        self.logger.info("\n".join(supr_message))

        report_progress(self)
        return WrappedResult(self.portrait, anchors)

    async def initialize_supertypes(self):
        """\
Initialize supertypes for all terms to start building portraits.
"""
        if self.portrait.ancestor_anchors:
            raise BouzygesError(
                "Should not happen: ancestor anchors are set, "
                "and yet initialize_supertypes is called"
            )

        supertypes_decode = {
            entry.term: entry.sctid for entry in self.__mrcm_entries
        }
        supertype_term: (
            SCTDescription | EscapeHatch
        ) = await self.prompter.prompt_supertype(
            term=self.source_term,
            options=supertypes_decode,
            allow_escape=False,
            term_context="; ".join(self.portrait.context)
            if self.portrait.context
            else None,
        )
        match supertype_term:
            case SCTDescription(answer_term):
                supertype = supertypes_decode[answer_term]
                self.logger.info(
                    f"Assuming {self.source_term} is {answer_term}"
                )
                self.portrait.ancestor_anchors.add(supertype)
            case EscapeHatch.WORD:
                raise BouzygesError(
                    "Should not happen: null-like response from prompter; "
                    "did the Prompter inject the escape hatch?"
                )

    async def populate_attribute_candidates(self) -> None:
        attributes: dict[
            SCTID, AttributeConstraints
        ] = await self.snowstorm.get_attribute_suggestions(
            self.portrait.ancestor_anchors
        )

        # Remove previously rejected attributes
        for attribute in self.portrait.rejected_attributes:
            attributes.pop(attribute, None)

        possible_message = []
        possible_message.append("Possible attributes for: " + self.source_term)
        for sctid, attribute in attributes.items():
            possible_message.append(f" - {sctid} {attribute.pt}")
        self.logger.debug("\n".join(possible_message))

        # Confirm the attributes
        for attribute in attributes.values():
            accept: bool = await self.prompter.prompt_attr_presence(
                term=self.source_term,
                attribute=attribute.pt,
                term_context="; ".join(self.portrait.context)
                if self.portrait.context
                else None,
            )
            self.logger.info(
                f"{attribute.sctid} {attribute.pt}: " + "Present"
                if accept
                else "Not present"
            )

            if accept:
                self.portrait.unchecked_attributes.add(attribute.sctid)

        # Remember the constraints
        for sctid, attribute in attributes.items():
            if sctid not in self.portrait.unchecked_attributes:
                continue
            self.portrait.relevant_constraints[sctid] = attribute

    async def populate_unchecked_attributes(self) -> None:
        rejected = set()
        for attribute in self.portrait.unchecked_attributes:
            self.logger.debug(f"Attribute: {attribute}")
            # Get possible attribute values
            values_options: dict[SCTID, SCTDescription] = (
                self.snowstorm.get_attribute_values(self.portrait, attribute)
            )
            self.logger.debug(f"Values: {values_options}")

            if not values_options:
                # No valid values for this attribute and parent combination
                rejected.add(attribute)
                continue
            else:
                possible_message = []
                possible_message.append(
                    f"Possible values for {attribute} in {self.source_term}"
                )
                for value in values_options:
                    possible_message.append(
                        f" - {value} {values_options[value]}"
                    )
                self.logger.debug("\n".join(possible_message))

            # Prompt for the value
            value_term: (
                SCTDescription | EscapeHatch
            ) = await self.prompter.prompt_attr_value(
                term=self.source_term,
                attribute=self.portrait.relevant_constraints[attribute].pt,
                options=values_options.values(),
                term_context="; ".join(self.portrait.context)
                if self.portrait.context
                else None,
                allow_escape=True,
            )

            match value_term:
                case SCTDescription(answer_term):
                    sctid = next(
                        SCTID(sctid)
                        for sctid, term in values_options.items()
                        if term == answer_term
                    )
                    self.portrait.attributes[attribute] = sctid
                case EscapeHatch.WORD:
                    # Choosing no attribute on initial prompt
                    # means rejection
                    rejected.add(attribute)

        self.portrait.rejected_attributes |= rejected
        # All are seen by now
        self.portrait.unchecked_attributes = set()

    async def update_existing_attr_values(self) -> None:
        """\
Update existing attribute values with the most precise descendant for all terms.
"""
        new_attributes = {}
        for attribute, value in self.portrait.attributes.items():
            new_attributes[attribute] = value
            while True:
                # Get children of the current value
                children = await self.snowstorm.get_concept_children(
                    new_attributes[attribute]
                )
                if not children:
                    # Leaf node
                    break

                descriptions = {v: k for k, v in children.items()}

                # Prompt for the most precise value
                value_term: (
                    SCTDescription | EscapeHatch
                ) = await self.prompter.prompt_attr_value(
                    term=self.source_term,
                    attribute=self.portrait.relevant_constraints[attribute].pt,
                    options=descriptions,
                    term_context="; ".join(self.portrait.context)
                    if self.portrait.context
                    else None,
                    allow_escape=True,
                )

                if isinstance(value_term, EscapeHatch):
                    # None of the children are correct
                    break

                new_attributes[attribute] = descriptions[value_term]

        self.portrait.attributes.update(new_attributes)

    async def update_anchors(self) -> bool:
        i = 0
        ancestors_changed = True
        while ancestors_changed:
            ancestors_changed = await self.__update_anchor()
            i += ancestors_changed
        self.logger.info(
            f"Updated {self.source_term} anchors in {i} iterations."
        )
        return i > 1

    async def __update_anchor(self) -> bool:
        """\
Update the ancestor anchors for one term to more precise children. Performs a
single iteration. Return True if the parent anchors have changed, False
otherwise.
"""
        new_anchors = set()

        async def process_anchor(anchor):
            # Get all immediate descendants
            children = await self.snowstorm.get_concept_children(anchor)
            self.logger.debug(f"Filtering {len(children)} children of {anchor}")

            # Remove verbatim known ancestors
            for known_ancestor in self.portrait.ancestor_anchors:
                children.pop(known_ancestor, None)

            # Filter previously rejected ancestors including meta-ancestors
            remaining = await self.snowstorm.filter_bad_descendants(
                children=children, bad_parents=self.portrait.rejected_supertypes
            )

            # Save the rejected children as rejected ancestors
            self.portrait.rejected_supertypes.update(
                child for child in children if child not in remaining
            )

            for child in set(children) - remaining:
                del children[child]

            self.logger.debug(f"Filtered to {len(children)}")

            # Iterate over descendants and ask LLM/Snowstorm if to include them
            # to the new anchors
            async def process_child(child_item):
                child_id, child_term = child_item

                child_concept = await self.snowstorm.get_concept(child_id)
                supertype: bool = (
                    await self.snowstorm.check_inferred_subsumption(
                        child_concept, self.portrait
                    )
                )

                if not supertype:
                    # This will prevent expensive re-queries on descendants
                    self.portrait.rejected_supertypes.add(child_id)
                    return

                # Primitive concepts must be confirmed by the LLM
                primitive = not child_concept.defined
                source_term_context = (
                    "; ".join(self.portrait.context)
                    if self.portrait.context
                    else None
                )
                if primitive:
                    if not await self.prompter.prompt_subsumption(
                        term=self.portrait.source_term,
                        prospective_supertype=child_term,
                        term_context=source_term_context,
                    ):
                        self.portrait.rejected_supertypes.add(child_id)
                    return

                new_anchors.add(child_id)

            asyncio.gather(*map(process_child, children.items()))

        asyncio.gather(*map(process_anchor, self.portrait.ancestor_anchors))

        if not new_anchors:
            self.logger.debug("No new ancestors found")
            return False

        # Update the anchor set with the new one
        self.logger.debug(
            f"New ancestors: {new_anchors - self.portrait.ancestor_anchors}"
        )
        self.portrait.ancestor_anchors |= new_anchors
        return True


# https://stackoverflow.com/questions/28655198/best-way-to-display-logs-in-pyqt
class BouzygesLoggingSpace(logging.Handler):
    """\
A logging handler that outputs log records to a QListView widget.
"""

    max_records = 2000

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.widget = QtWidgets.QListView(*args, **kwargs)
        self.model = QtCore.QStringListModel()
        self.widget.setModel(self.model)
        self.widget.setWordWrap(True)
        self.widget.setAlternatingRowColors(True)
        self.widget.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.setFormatter(_formatter)

    def emit(self, record):
        msg = self.format(record)

        current_count = self.model.rowCount()
        self.model.insertRow(current_count)
        index = self.model.index(current_count)
        self.model.setData(index, msg)

        # Scroll to the top
        slider = self.widget.verticalScrollBar()
        if slider:
            slider.setValue(slider.maximum())

        # Limit the number of records
        if self.model.rowCount() > self.max_records:
            self.model.removeRow(0)


class EnvironmentVariableEditor(QtWidgets.QDialog):
    """\
A dialog to edit environment variables.
"""

    def __init__(
        self, parent: BouzygesWindow, variables: EnvironmentParameters
    ) -> None:
        super().__init__(parent)
        self.logger = parent.logger.getChild(self.__class__.__name__)
        self.variables = variables
        self.parent_window = parent

        self.setWindowTitle("Override Environment variables")
        self.__dict = {}
        master_layout = QtWidgets.QVBoxLayout()
        warning_label = QtWidgets.QLabel(
            "<b>Warning</b>: Environment variables will not persist between "
            "sessions. For security reasons, they are neither logged nor "
            "saved in config JSON file. Use <code>.env</code> file for "
            "permanent changes."
        )
        warning_label.setWordWrap(True)
        master_layout.addWidget(warning_label)

        for envvar, value in variables.model_dump().items():
            self.__dict[envvar] = value
            layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(envvar.upper())
            label.setFixedWidth(200)
            layout.addWidget(label)
            edit = QtWidgets.QLineEdit()
            edit.setText(value)
            edit.setPlaceholderText("Unset")
            edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.PasswordEchoOnEdit)
            edit.setMinimumWidth(300)
            layout.addWidget(edit)
            master_layout.addLayout(layout)
        self.setLayout(master_layout)

        buttons_layout = QtWidgets.QHBoxLayout()
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save)

        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(save_button)
        master_layout.addLayout(buttons_layout)

    def save(self):
        for envvar, value in self.__dict.items():
            if value:
                logging.debug(f"Setting {envvar} to a new value")
            else:
                logging.debug(f"Unsetting {envvar}")
            setattr(self.variables, envvar, value or None)
            # Set them to os module to make sure they are available
            os.environ[envvar] = value or ""
        self.parent_window.reset_ui()
        self.accept()


class BouzygesWindow(QtWidgets.QMainWindow):
    """\
Main window and start config for the Bouzyges system.
"""

    def __init__(self, loop: asyncio.AbstractEventLoop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = LOGGER.getChild("GUI")
        self.setWindowTitle("OHDSI Bouzyges")
        self.setWindowIcon(QtGui.QIcon("icon.png"))

        layout = QtWidgets.QVBoxLayout()
        self.__populate_layout(layout)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Menu bar
        menubar = self.menuBar()
        if menubar is not None:
            self.populate_menu(menubar)

        # Async support
        self.loop = loop

    def __populate_layout(self, layout):
        self._verticalSpacer = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )

        # Options
        self.options_container = QtWidgets.QWidget()
        self.options_container.setMaximumWidth(350)
        options_layout = QtWidgets.QVBoxLayout()
        options_subtitle = QtWidgets.QLabel("Options")
        options_subtitle.setStyleSheet("font-weight: bold;")
        options_layout.addWidget(options_subtitle)
        options_contents = QtWidgets.QVBoxLayout()
        self.__populate_option_contents(options_contents)
        options_layout.addLayout(options_contents)
        spacer = QtWidgets.QSpacerItem(
            20,
            40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        options_layout.addItem(spacer)
        self.options_container.setLayout(options_layout)

        # Input and output
        right_quarter_layout = QtWidgets.QVBoxLayout()
        self.io_container = QtWidgets.QWidget()
        io_layout = QtWidgets.QVBoxLayout()
        self.__populate_io_layout(io_layout)
        self.io_container.setLayout(io_layout)
        right_quarter_layout.addWidget(self.io_container)

        # Run layout
        run_layout = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self.spin_bouzyges)
        self.run_status = QtWidgets.QLabel("Status: Ready")
        self.run_status.setStyleSheet("font-weight: bold;")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setDisabled(True)

        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.run_status)
        run_layout.addWidget(self.progress_bar)
        right_quarter_layout.addLayout(run_layout)
        # Logging space
        logging_layout = QtWidgets.QVBoxLayout()
        logging_subtitle = QtWidgets.QLabel("Run log")
        logging_subtitle.setStyleSheet("font-weight: bold;")
        self.log_display = BouzygesLoggingSpace()

        LOGGER.addHandler(self.log_display)

        self.logger.info("Logging to GUI is initialized")
        log_widget = self.log_display.widget
        logging_layout.addWidget(logging_subtitle)
        logging_layout.addWidget(log_widget)

        top_half_layout = QtWidgets.QHBoxLayout()
        top_half_layout.addWidget(self.options_container)
        top_half_layout.addLayout(right_quarter_layout)

        layout.addLayout(top_half_layout)
        layout.addLayout(logging_layout)

    def __populate_io_layout(self, layout) -> None:
        # Input file selection
        input_layout = QtWidgets.QVBoxLayout()
        input_subtitle = QtWidgets.QLabel("Input file")
        input_subtitle.setStyleSheet("font-weight: bold;")
        input_layout.addWidget(input_subtitle)
        input_doc = QtWidgets.QLabel()
        input_doc.setText(FileReader.input_doctext)
        input_layout.addWidget(input_doc)
        input_contents = QtWidgets.QHBoxLayout()
        self.input_file = QtWidgets.QLineEdit()
        self.input_file.setPlaceholderText("Select input CSV file")
        self.input_file.setText(PARAMS.read.file)
        self.input_file.setReadOnly(True)
        input_contents.addWidget(self.input_file)
        input_select = QtWidgets.QPushButton("Select")
        input_select.clicked.connect(self.select_input)
        input_contents.addWidget(input_select)
        input_layout.addLayout(input_contents)
        self.input_options_widget = IOParametersWidget(PARAMS.read, "Read")
        input_layout.addWidget(self.input_options_widget)

        # Output selection
        output_layout = QtWidgets.QVBoxLayout()
        output_subtitle = QtWidgets.QLabel("Output file")
        output_subtitle.setStyleSheet("font-weight: bold;")
        output_layout.addWidget(output_subtitle)
        out_dir_contents = QtWidgets.QHBoxLayout()
        out_dir_label = QtWidgets.QLabel("Output directory:")
        out_dir_contents.addWidget(out_dir_label)
        self.output_dir = QtWidgets.QLineEdit()
        self.output_dir.setPlaceholderText("Select output directory")
        self.output_dir.setText(os.getcwd())
        self.output_dir.setReadOnly(True)
        out_dir_contents.addWidget(self.output_dir)
        output_select = QtWidgets.QPushButton("Select")
        output_select.clicked.connect(self.select_output)
        out_dir_contents.addWidget(output_select)

        out_file_contents = QtWidgets.QHBoxLayout()
        out_file_label = QtWidgets.QLabel("File name:")
        out_file_contents.addWidget(out_file_label)
        output_filename = QtWidgets.QLineEdit()
        self.out_options_widget = IOParametersWidget(PARAMS.write, "Write")
        output_filename.setPlaceholderText("Output file name")
        output_filename.setText(PARAMS.write.file)
        output_filename.textChanged.connect(self.out_options_widget.update_file)
        out_file_contents.addWidget(output_filename)
        out_format_label = QtWidgets.QLabel("Format:")
        out_file_contents.addWidget(out_format_label)
        out_format_select = QtWidgets.QComboBox()
        out_format_select.addItems(FileWriter.get_formats())
        out_format_select.setCurrentIndex(
            FileWriter.get_formats().index(PARAMS.format)
        )
        out_format_select.currentIndexChanged.connect(self.format_changed)
        self.out_options_widget.setEnabled(PARAMS.format != "JSON")
        out_file_contents.addWidget(out_format_select)

        output_layout.addLayout(out_dir_contents)
        output_layout.addLayout(out_file_contents)
        output_layout.addWidget(self.out_options_widget)

        layout.addLayout(input_layout)
        layout.addItem(self._verticalSpacer)
        layout.addLayout(output_layout)
        layout.addItem(self._verticalSpacer)

    def format_changed(self, idx: int) -> None:
        PARAMS.format = FileWriter.get_formats()[idx]
        self.out_options_widget.setEnabled(PARAMS.format != "JSON")
        self.logger.info(f"Output format changed to {PARAMS.format}")

    def select_input(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select input CSV file",
        )
        if file:
            self.input_options_widget.update_file(
                file[0], self.input_file.setText
            )

    def select_output(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select directory for output",
        )
        if dir:
            self.output_dir.setText(dir)
            PARAMS.out_dir = dir
            self.logger.info(f"Output directory changed to {dir}")

    def select_cache_db(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select cache database file",
        )
        if file:
            PARAMS.api.cache_db = file[0]

    @qasync.asyncSlot()
    async def spin_bouzyges(self) -> None:
        # Adding a file handler to the logger
        if PARAMS.log.log_to_file:
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f"bouzyges-{date_str}.log"
            log_file = os.path.join(self.output_dir.text(), file_name)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(_formatter)

            LOGGER.addHandler(file_handler)
            self.logger.info(f"Now logging to file: {log_file}")

        self.options_container.setEnabled(False)
        self.io_container.setEnabled(False)
        self.run_button.setEnabled(False)
        bouzyges_capture: dict[Literal["success"], Bouzyges] = {}
        bouzyges: Bouzyges

        async def prepare(progress_callback):
            try:
                bouzyges_capture["success"] = await Bouzyges.prepare(
                    progress_callback=progress_callback
                )
            except Exception as e:
                self.logger.error(f"Could not prepare Bouzyges: {e}")
                self.logger.error(
                    "Could not prepare Bouzyges. Check the configuration."
                )
                self.reset_ui()
                return

        await self.__start_job("Bouzyges Preparation", prepare)

        bouzyges = bouzyges_capture["success"]
        await self.__start_job("Bouzyges Run", bouzyges.run)

        await self.__start_job("Bouzyges Output", bouzyges.output)

        self.reset_ui()

    def reset_ui(self) -> None:
        self.input_options_widget.set_values()
        self.out_options_widget.set_values()
        self.options_container.setEnabled(True)
        self.io_container.setEnabled(True)
        self.run_button.setEnabled(True)
        self.run_status.setText("Status: Ready")
        self.progress_bar.setValue(0)
        self.progress_bar.setEnabled(False)

    def update_progress(self, value: int, max: int) -> None:
        self.progress_bar.setRange(0, max)
        self.progress_bar.setValue(value)

    async def __start_job(self, name: str, async_target, *args, **kwargs):
        self.logger.info(f"Starting {name} thread")
        self.run_status.setText(f"Status: {name}")
        self.progress_bar.setEnabled(True)
        result = await async_target(
            *args, **kwargs, progress_callback=self.update_progress
        )
        self.logger.info(f"{name} thread finished")
        return result

    def __populate_option_contents(self, layout) -> None:
        # Prompter choice:
        prompter_layout = QtWidgets.QVBoxLayout()

        ## Prompter imlementations
        impl_layout = QtWidgets.QHBoxLayout()
        impl_label = QtWidgets.QLabel("Prompter:")
        impl_label.setStyleSheet("font-weight: bold;")
        impl_options = QtWidgets.QComboBox()
        impl_options.addItems(AVAILABLE_PROMPTERS.values())
        current_impl_idx = list(AVAILABLE_PROMPTERS).index(PARAMS.api.prompter)
        impl_options.setCurrentIndex(current_impl_idx)
        impl_options.currentIndexChanged.connect(self.prompter_changed)
        impl_layout.addWidget(impl_label)
        impl_layout.addWidget(impl_options)

        ## Model id
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model ID:")
        model_input = QtWidgets.QLineEdit()
        model_input.setPlaceholderText(DEFAULT_MODEL)
        model_input.setText(PARAMS.api.llm_model_id)
        model_input.textChanged.connect(self.model_id_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_input)

        prompter_layout.addLayout(impl_layout)
        prompter_layout.addLayout(model_layout)

        # Snowstorm connection options
        snowstorm_layout = QtWidgets.QVBoxLayout()
        snowstorm_subtitle = QtWidgets.QLabel("Snowstorm API:")
        snowstorm_subtitle.setStyleSheet("font-weight: bold;")
        snowstorm_layout.addWidget(snowstorm_subtitle)
        snowstorm_contents = QtWidgets.QVBoxLayout()
        snowstorm_url_layout = QtWidgets.QHBoxLayout()
        snowstorm_url_label = QtWidgets.QLabel("Endpoint URL:")
        snowstorm_url_input = QtWidgets.QLineEdit()
        snowstorm_url_input.setPlaceholderText("http://localhost:8080/")
        snowstorm_url_input.setText(PARAMS.api.snowstorm_url)
        snowstorm_url_input.textChanged.connect(self.snowstorm_url_changed)
        snowstorm_url_layout.addWidget(snowstorm_url_label)
        snowstorm_url_layout.addWidget(snowstorm_url_input)
        snowstorm_contents.addLayout(snowstorm_url_layout)
        snowstorm_layout.addLayout(snowstorm_contents)

        # Sqlite database options
        sqlite_layout = QtWidgets.QVBoxLayout()
        sqlite_subtitle = QtWidgets.QLabel(
            "Prompt cache path (set to empty to disable):"
        )
        sqlite_subtitle.setStyleSheet("font-weight: bold;")
        sqlite_db_file = QtWidgets.QLineEdit()
        sqlite_db_file.setPlaceholderText("None")
        sqlite_db_file.setText(PARAMS.api.cache_db)
        sqlite_db_file.textChanged.connect(self.cache_db_changed)
        sqlite_select = QtWidgets.QPushButton("Select")
        sqlite_select.clicked.connect(self.select_cache_db)
        sqlite_layout.addWidget(sqlite_subtitle)
        sqlite_selector_layout = QtWidgets.QHBoxLayout()
        sqlite_selector_layout.addWidget(sqlite_db_file)
        sqlite_selector_layout.addWidget(sqlite_select)
        sqlite_layout.addLayout(sqlite_selector_layout)

        # Developer options
        prof_label = QtWidgets.QLabel("Profiling:")
        prof_label.setStyleSheet("font-weight: bold;")
        prof_layout = QtWidgets.QHBoxLayout()

        early_termination_label = QtWidgets.QLabel("Stop after (s):")
        early_termination_input = QtWidgets.QLineEdit()
        early_termination_input.setPlaceholderText("don't")
        if PARAMS.prof.stop_profiling_after_seconds is not None:
            early_termination_input.setText(
                str(PARAMS.prof.stop_profiling_after_seconds)
            )
        early_termination_input.textChanged.connect(self.et_changed)

        profiling_layout = QtWidgets.QVBoxLayout()
        profiling_checkbox = QtWidgets.QCheckBox("Generate stats.prof")
        profiling_checkbox.setChecked(PARAMS.prof.enabled)
        profiling_checkbox.stateChanged.connect(self.profiling_changed)
        profiling_layout.addWidget(profiling_checkbox)

        prof_layout.addWidget(early_termination_label)
        prof_layout.addWidget(early_termination_input)
        prof_layout.addLayout(profiling_layout)

        logging_label = QtWidgets.QLabel("Logging:")
        logging_label.setStyleSheet("font-weight: bold;")
        logging_layout = QtWidgets.QHBoxLayout()

        log_to_file_checkbox = QtWidgets.QCheckBox("Log to file")
        log_to_file_checkbox.setChecked(PARAMS.log.log_to_file)
        log_to_file_checkbox.stateChanged.connect(self.ltf_changed)

        logging_level_label = QtWidgets.QLabel("Logging level:")
        logging_level_options = QtWidgets.QComboBox()
        logging_level_options.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        current_logging_level_idx = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ].index(PARAMS.log.logging_level)
        logging_level_options.setCurrentIndex(current_logging_level_idx)
        logging_level_options.currentIndexChanged.connect(self.ll_changed)

        logging_layout.addWidget(logging_level_label)
        logging_layout.addWidget(logging_level_options)
        logging_layout.addWidget(log_to_file_checkbox)

        layout.addLayout(prompter_layout)
        layout.addLayout(snowstorm_layout)
        layout.addLayout(sqlite_layout)
        layout.addWidget(prof_label)
        layout.addLayout(prof_layout)
        layout.addWidget(logging_label)
        layout.addLayout(logging_layout)

    def ltf_changed(self, state) -> None:
        PARAMS.log.log_to_file = state == 2
        self.logger.debug(f"Logging to file changed to: {state == 2}")

    def prompter_changed(self, index) -> None:
        new_prompter = list(AVAILABLE_PROMPTERS)[index]
        PARAMS.api.prompter = new_prompter
        self.logger.debug(f"Prompter changed to: {new_prompter}")

    def profiling_changed(self, state) -> None:
        PARAMS.prof.enabled = state == 2
        self.logger.debug(f"Profiling changed to: {state == 2}")

    def et_changed(self, text) -> None:
        try:
            new_et = int(text)
        except ValueError:
            PARAMS.prof.stop_profiling_after_seconds = None
            self.logger.debug("Invalid input for early termination, resetting")
            return

        if new_et <= 0:
            new_et = None

        PARAMS.prof.stop_profiling_after_seconds = new_et
        self.logger.debug(f"Early termination changed to: {new_et}")

    def ll_changed(self, index) -> None:
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
        new_level = levels[index]
        self.logger.debug(f"Logging level changed to: {new_level}")
        PARAMS.log.update(new_level)

    def snowstorm_url_changed(self, text) -> None:
        if not text:
            text = "http://localhost:8080/"
        PARAMS.api.snowstorm_url = Url(text)
        self.logger.debug(f"Snowstorm URL changed to: {text}")

    def model_id_changed(self, text) -> None:
        if not text:
            text = DEFAULT_MODEL

        PARAMS.api.llm_model_id = text
        self.logger.debug(f"LLM model_id set to: {text}")

    def cache_db_changed(self, text) -> None:
        if not text:
            text = None
        PARAMS.api.cache_db = text
        self.logger.debug(f"Cache database path changed to: {text}")

    def populate_menu(self, menubar: QtWidgets.QMenuBar) -> None:
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        help_menu = menubar.addMenu("Help")

        if file_menu is None or edit_menu is None or help_menu is None:
            raise RuntimeError("Could not create menu bar")

        # File menu items
        load_config_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("document-open"),
            text="Load configuration",
            parent=self,
        )
        load_config_action.triggered.connect(self.load_config)
        load_config_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_config_action)

        save_config_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("document-save"),
            text="Save configuration",
            parent=self,
        )
        save_config_action.triggered.connect(self.save_config)
        save_config_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_config_action)

        quit_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("application-exit"),
            text="Quit",
            parent=self,
        )
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu items
        edit_env_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("preferences-system"),
            text="Override environment variables",
            parent=self,
        )
        edit_env_action.triggered.connect(
            lambda: EnvironmentVariableEditor(self, PARAMS.env).exec()
        )
        edit_menu.addAction(edit_env_action)

        # Help menu items
        about_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("help-about"), text="About", parent=self
        )
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

        license_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("help-about"),
            text="License",
            parent=self,
        )
        license_action.triggered.connect(self.license)
        help_menu.addAction(license_action)

        report_issue_action = QtGui.QAction(
            icon=QtGui.QIcon.fromTheme("help-report-bug"),
            text="Report an issue or get help",
            parent=self,
        )
        report_issue_action.triggered.connect(self.report_issue)
        report_issue_action.setShortcut("F1")
        help_menu.addAction(report_issue_action)

    def about(self):
        label = (
            "Bouzyges is a tool for identifying the most specific "
            "ancestors of a set of terms in the SNOMED CT ontology."
            "\n\n"
            "All information is available at the project's GitHub page."
        )
        QtWidgets.QMessageBox.about(self, "About Bouzyges", label)

    def report_issue(self):
        webbrowser.open("https://github.com/OHDSI/Bouzyges")

    def license(self):
        # TODO: Add license
        QtWidgets.QMessageBox.about(
            self,
            "License",
            "Bouzyges is not currently licensed for distribution, nor for use. "
            "If you are using it now, consider yourself a developer. All risks "
            "are yours.",
        )

    def load_config(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select JSON configuration file",
        )
        if file:
            with open(file[0], "r") as f:
                json_config = json.load(f)
                PARAMS.update(json_config)
                self.reset_ui()

            self.logger.info(f"Configuration loaded from {file[0]}")

    def save_config(self):
        file = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save JSON configuration file",
        )
        if file:
            with open(file[0], "w") as f:
                json.dump(PARAMS.model_dump(), f, indent=2)

            self.logger.info(f"Configuration saved to {file[0]}")


def main():
    loop = qasync.QEventLoop()
    asyncio.set_event_loop(loop)
    window = BouzygesWindow(loop=loop)
    window.show()
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    APP = qasync.QApplication(sys.argv)
    main()
