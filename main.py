import cProfile
import datetime
import itertools
import json
import logging
import openai
import os
import pstats
import re
import requests
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from frozendict import frozendict
from typing import Callable, Iterable, Literal, Mapping, TypeVar

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


# Parameters
# TODO: use sys.argv for these
PROFILING = False
logging.basicConfig(level=logging.DEBUG)
STOP_AFTER_SECONDS: int | None = 600


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


class URL(str):
    """REST URL string"""


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


JsonPrimitive = int | float | str | bool | None
OpenAIPromptRole = Literal["user", "system", "assisstant"]
OpenAIMessages = tuple[frozendict[OpenAIPromptRole, str]]
T = TypeVar("T")

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


# SNOMED root concept
ROOT_CONCEPT = SCTID(138875005)


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
    guide_link: URL  # For eventual RAG connection
    # Currently uses unparseable extension of ECL grammar,
    # but one day we will use it
    domain_template: ECLExpression
    parent_domain: ECLExpression | None = None
    proximal_primitive_refinement: ECLExpression | None = None

    @classmethod
    def from_json(cls, json_data: dict):
        af = json_data["additionalFields"]
        pd = af.get("parentDomain")
        prf = af.get("proximalPrimitiveRefinement")

        return cls(
            sctid=SCTID(json_data["referencedComponent"]["conceptId"]),
            term=SCTDescription(json_data["referencedComponent"]["pt"]["term"]),
            domain_template=ECLExpression(
                # Use precoordination: stricter
                af["domainTemplateForPrecoordination"]
            ),
            domain_constraint=ECLExpression(af["domainConstraint"]),
            guide_link=URL(af["guideURL"]),
            parent_domain=ECLExpression(pd) if pd else None,
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

    def __init__(self, term: str, context: str | None = None) -> None:
        self.source_term: str = term
        self.context: Iterable[str] | None = context
        self.ancestor_anchors: set[SCTID] = set()

        self.unchecked_attributes: set[SCTID] = set()
        self.attributes: dict[SCTID, SCTID] = {}
        self.rejected_attributes: set[SCTID] = set()
        self.rejected_supertypes: set[SCTID] = set()

        self.relevant_constraints: dict[SCTID, AttributeConstraints] = {}


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
        logging.error("Request:")
        logging.error(response.request.method, response.request.url)
        if response:
            logging.error("Response:")
            logging.error(json.dumps(response.json(), indent=2))
        return cls(
            f"Snowstorm API returned {response.status_code} status code",
            response,
        )


class PrompterError(Exception):
    """Raised when the prompter encounters an error."""


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


# Logic classes
## Prompt cache interface
class PromptCache:
    """\
Interface for a prompt cache.

Saves prompts and answers to avoid re-prompting the same questions and wasting
tokens.
"""

    def __init__(self, db_connection: sqlite3.Connection):
        self.connection = db_connection
        self.table_name = "prompt"

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
        query = f"""
            SELECT response
            FROM {self.table_name}
            WHERE
                model = ? AND
                prompt_text = ? AND
                prompt_is_json = ? AND
                api_options
        """

        # Shape the prompt text
        if prompt.api_options:
            api_options = json.dumps(
                {
                    key: prompt.api_options[key]
                    for key in sorted(prompt.api_options)
                }
            )
            query += " = ? ;"
        else:
            api_options = None
            query += " is ? ;"

        prompt_is_json = not isinstance(prompt.prompt_message, str)
        prompt_text = json.dumps(prompt.prompt_message)

        cursor = self.connection.cursor()
        cursor.execute(
            query,
            (model, prompt_text, prompt_is_json, api_options),
        )
        if answer := cursor.fetchone():
            return answer[0]
        return None

    def remember(self, model: str, prompt: Prompt, response: str) -> None:
        """\
Remember the answer for the prompt for the specified model.
"""
        query = f"""
            INSERT INTO {self.table_name} (
                model, prompt_text, prompt_is_json, response, api_options
            )
            VALUES (?, ?, ?, ?)
        """
        if prompt.api_options:
            api_options = json.dumps(
                {
                    key: prompt.api_options[key]
                    for key in sorted(prompt.api_options)
                }
            )
        else:
            api_options = None
        cursor = self.connection.cursor()

        # Shape the prompt text
        if prompt_is_json := isinstance(prompt.prompt_message, list):
            prompt_text = json.dumps(prompt.prompt_message)
        else:
            prompt_text = prompt.prompt_message

        cursor.execute(
            query,
            (
                model,
                prompt_text,
                prompt_is_json,
                response,
                api_options,
            ),
        )
        self.connection.commit()


## Logic prompt format classes
class PromptFormat(ABC):
    """\
Abstract class for formatting prompts for the LLM agent.
"""

    ROLE = (
        "a domain expert in clinical terminology who is helping to build a "
        "semantic portrait of a concept in a clinical terminology system"
    )
    TASK = (
        "to provide information about the given term supertypes, "
        "attributes, attribute values, and other relevant information as "
        "requested"
    )
    REQUIREMENTS = (
        "in addition to providing accurate factually correct information, "
        "it is critically important that you provide answer in a "
        "format that is requested by the system, as answers will "
        "be parsed by a machine. Your answer should ALWAYS end with a line "
        "that says 'The answer is ' and the chosen option"
    )
    INSTRUCTIONS = (
        "Options that speculate about details not explicitly included in the"
        "term meaning are to be avoided, e.g. term 'operation on abdominal "
        "region' should not be assumed to be a laparoscopic operation, as "
        "access method is not specified in the term. It is encouraged to "
        "explain your reasoning when providing answers. The automated system "
        "will look for the last answer surrounded by square brackets, e.g. "
        "[answer], so only one of the options should be selected and returned "
        "in this format. If the question looks like 'What is the topography of "
        "the pulmonary tuberculosis?', and the options are [Lung structure], "
        "[Heart structure], [Kidney structure], the good answer would look "
        "like 'As the term 'pulmonary' refers to a disease of the lungs, "
        "the topography should be [Lung structure].' If you are not sure about "
        "the answer, you are encouraged to think aloud, analyzing the options."
    )

    ESCAPE_INSTRUCTIONS = (
        " If all provided options are incorrect, or imply extra information "
        "not present in the term, you must explain why each option is "
        "incorrect, and finalize the answer with the word "
        f"{EscapeHatch.WORD}."
    )

    ROLE: str
    TASK: str
    REQUIREMENTS: str
    INSTRUCTIONS: str
    ESCAPE_INSTRUCTIONS: str

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
            f"'{term}'."
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
            f"Is the term '{term}' a subtype of the concept "
            f"'{prospective_supertype}'?"
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
        presence_penalty=-0.5,
        max_tokens=1024,
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
            history,  # type:ignore
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
                "Your exact instructions are:",
            ),
            (
                "system",
                self.INSTRUCTIONS + (allow_escape * self.ESCAPE_INSTRUCTIONS),
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
                    ("Attribute is defined as follows: " + attribute_context),
                )
            )

        prompt.append(
            (
                "user",
                f"""Options are:
                    - {BooleanAnswer.YES}: The attribute is guaranteed present.
                    - {BooleanAnswer.NO}: The attribute is absent or not guaranteed present.
                    """,
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
                    f"Choose the value of the attribute '{attribute}' in the term "
                    f"'{term}'."
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
                    - {BooleanAnswer.YES}: The attribute is guaranteed present.
                    - {BooleanAnswer.NO}: The attribute is absent or not guaranteed present.
                    """,
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
        use_cache: bool = True,
        **kwargs,
    ):
        _ = args, kwargs
        self.prompt_format = prompt_format
        self.cache: PromptCache | None = None
        if use_cache:
            self.cache = PromptCache(sqlite3.connect("prompt_cache.db"))

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

        if not options:
            # Return the answer in brackets, if there is one
            if last_line.count("[") == last_line.count("]") == 1:
                start = last_line.index("[") + 1
                end = last_line.index("]")
                return SCTDescription(last_line[start:end])
            else:
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

        counts = {}
        for option in wrapped_options:
            counts[option] = last_line.count(option)

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
            option: last_line.rfind(wrapped)
            for wrapped, option in wrapped_options.items()
        }
        if all(index == -1 for index in indices.values()):
            raise PrompterError(
                "Could not find a unique option in the answer:", last_line
            )

        return max(indices, key=lambda k: indices.get(k, -1))

    @staticmethod
    def unwrap_bool_answer(
        answer: str,
        yes: str = BooleanAnswer.YES,
        no: str = BooleanAnswer.NO,
    ) -> bool:
        """\
Check if the answer contains a yes or no option.
"""
        words = answer.strip().splitlines()[-1].split()
        if yes in words and no not in words:
            return True
        elif no in words and yes not in words:
            return False
        else:
            raise PrompterError(
                "Could not find an unambiguous boolean answer in the response"
            )

    def prompt_supertype(
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
        logging.debug("Constructed prompt: %s", prompt.prompt_message)

        # Try the cache
        if self.cache:
            if cached_answer := self.cache.get(self._model_id, prompt):
                answer = self.unwrap_class_answer(
                    cached_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
                logging.info(f"From cache: {answer} is a supertype of {term}")
                return answer

        # Get the answer
        answer = self._prompt_class_answer(allow_escape, options, prompt)
        logging.info(f"Agent answer: {answer} is a supertype of {term}")
        return answer

    def prompt_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> bool:
        prompt: Prompt = self.prompt_format.form_attr_presence(
            term, attribute, term_context, attribute_context
        )

        if self.cache:
            if cached_answer := self.cache.get(self._model_id, prompt):
                answer = self.unwrap_bool_answer(cached_answer)
                logging.info(
                    f"From cache: The attribute '{attribute}' is "
                    f"{'present' if answer else 'absent'} in '{term}'"
                )
                return answer

        answer = self._prompt_bool_answer(prompt)
        logging.info(
            f"Agent answer: The attribute '{attribute}' is "
            f"{'present' if answer else 'absent'} in '{term}'"
        )
        return answer

    def prompt_attr_value(
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

        if self.cache:
            if cached_answer := self.cache.get(self._model_id, prompt):
                answer = self.unwrap_class_answer(
                    cached_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
                logging.info(
                    f"From cache: The value of the attribute '{attribute}' in "
                    f"'{term}' is '{answer}'"
                )
                return answer

        answer = self._prompt_class_answer(allow_escape, options, prompt)
        logging.info(
            f"Agent answer: The value of the attribute '{attribute}' in "
            f"'{term}' is '{answer}'"
        )
        return answer

    def prompt_subsumption(
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

        if self.cache:
            if cached_answer := self.cache.get(self._model_id, prompt):
                answer = self.unwrap_bool_answer(cached_answer)
                logging.info(
                    f"From cache: The term '{term}' is",
                    "a subtype" if answer else "not a subtype",
                    f"of '{prospective_supertype}'",
                )
                return answer

        answer = self._prompt_bool_answer(prompt)
        logging.info(
            f"Agent answer: The term '{term}' is",
            "a subtype" if answer else "not a subtype",
            f"of '{prospective_supertype}'",
        )
        return answer

    def cache_remember(self, prompt: Prompt, answer: str) -> None:
        if self.cache:
            self.cache.remember(self._model_id, prompt, answer)

    # Following methods are abstract and represent common queries to the model
    @abstractmethod
    def _prompt_bool_answer(self, prompt: Prompt) -> bool:
        """\
Send a prompt to the counterpart agent to obtain the answer
"""

    @abstractmethod
    def _prompt_class_answer(
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

    def _prompt_class_answer(self, allow_escape, options, prompt):
        while True:
            print(prompt.prompt_message)
            brain_answer = input("Answer: ").strip()
            try:
                answer = self.unwrap_class_answer(
                    brain_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
                self.cache_remember(prompt, brain_answer)
                return answer

            except PrompterError as e:
                print("Error:", e)
                print("Please, provide an answer in the requested format.")

    def _prompt_bool_answer(self, prompt: Prompt) -> bool:
        while True:
            print(prompt.prompt_message)
            brain_answer = input("Answer: ").strip()
            try:
                answer = self.unwrap_bool_answer(brain_answer)
                self.cache_remember(prompt, brain_answer)
                return answer
            except PrompterError as e:
                print("Error:", e)
                print("Please, provide an answer in the requested format.")

    def ping(self) -> bool:
        input("Hey, are you here? (Press Enter) ")
        print("Good human.")
        return True

    def report_usage(self) -> None:
        logging.info(
            "No usage to report, as this is a human prompter. Stay "
            "hydrated and have a good day!"
        )


class OpenAIAzurePrompter(Prompter):
    """\
A prompter that interfaces with the OpenAI API using Azure.
"""

    DEFAULT_MODEL = "gpt-35-turbo"
    DEFAULT_VERSION = "2024-06-01"

    ATTEMPTS_BEFORE_FAIL = 3

    def __init__(
        self,
        *args,
        api_key: str,
        azure_endpoint: str,
        api_version: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        logging.info("Initializing the Azure API client...")
        self._model_id = model or self.DEFAULT_MODEL
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=self._model_id,
            api_version=api_version or self.DEFAULT_VERSION,
        )

        self._token_usage: dict[Prompt, int] = {}
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.encoding_for_model(self._model_id)
            except KeyError:
                logging.warning("Model not found in the token")
                self._encoding = None
        else:
            logging.warning("Tiktoken not installed, can't track token usage")
            self._encoding = None

    def ping(self) -> bool:
        logging.info("Pinging the Azure API...")
        # Ping by retrieving the list of models
        headers = {"api-key": self._client.api_key}
        try:
            models = self._client.models.list(extra_headers=headers, timeout=5)
        except openai.APITimeoutError:
            logging.warning("Connection timed out")
            return False
        response: dict = models.model_dump()
        success = response.get("data", []) != []
        if success:
            logging.info("API is available")
            if self._model_id not in [obj["id"] for obj in response["data"]]:
                logging.warning(
                    f"But '{self._model_id}' is not present in the API response!"
                )
                return False
            return True

        logging.warning("API is not available")
        return False

    def _prompt_bool_answer(self, prompt: Prompt) -> bool:
        return self._prompt_answer(prompt, self.unwrap_bool_answer)

    def _prompt_class_answer(self, allow_escape, options, prompt):
        return self._prompt_answer(
            prompt,
            lambda x: self.unwrap_class_answer(
                x, options, EscapeHatch.WORD if allow_escape else None
            ),
        )

    def _prompt_answer(self, prompt: Prompt, parser: Callable[[str], T]) -> T:
        logging.info("Prompting the Azure API for an answer...")

        if self._encoding:
            token_count = len(
                self._encoding.encode(json.dumps(prompt.prompt_message))
            )
            self._token_usage[prompt] = token_count
            logging.debug("Token usage:", token_count)
        else:
            logging.warning("Token usage will not be calculated: unknown model")

        for attempt in range(self.ATTEMPTS_BEFORE_FAIL + 1):
            logging.debug("Prompt message", prompt.prompt_message)
            if attempt > 0:
                logging.warning("Retrying...")

            if not isinstance(prompt.prompt_message, str):
                # Clean-up types for plain JSON
                prompt_text = json.loads(json.dumps(prompt.prompt_message))
            else:
                prompt_text = prompt.prompt_message

            try:
                brain_answer = self._client.completions.create(
                    model=self._model_id,
                    prompt=prompt_text,
                    # Prefer answers with same tokens
                    **(prompt.api_options or {}),
                )
            except openai.APIError as e:
                logging.error("API error:", e)
                if attempt == self.ATTEMPTS_BEFORE_FAIL:
                    raise PrompterError("Failed to get a response from the API")
                continue

            try:
                answer = parser(brain_answer.choices[0].text)
                self.cache_remember(prompt, brain_answer.choices[0].text)
                return answer
            except PrompterError as e:
                logging.error("Error:", e)
                if attempt == self.ATTEMPTS_BEFORE_FAIL:
                    raise PrompterError("Failed to parse the answer")
                continue

        # Unreachable
        raise PrompterError("Unreachable code reached?!")

    def report_usage(self) -> None:
        logging.info("Reporting usage to the Azure API...")
        if self._token_usage:
            n_prompts = len(self._token_usage)
            total_tokens = sum(self._token_usage.values())
            logging.info(
                f"Reporting {n_prompts} prompts with a total of "
                f"{total_tokens} tokens"
            )
        else:
            logging.warning("No token usage to report")


class SnowstormAPI:
    TARGET_CODESYSTEM = "SNOMEDCT"
    CONTENT_TYPE_PREFERENCE = "NEW_PRECOORDINATED", "PRECOORDINATED", "ALL"
    PAGINATION_STEP = 100
    MAX_BAD_PARENT_QUERY = 32

    def __init__(self, url: URL):
        # Debug
        self.__start_time = datetime.datetime.now()

        self.url: URL = url
        self.branch_path: BranchPath = self.get_main_branch_path()

        # Cache repetitive queries
        self.__concepts_cache: dict[SCTID, Concept] = {}
        self.__subsumptions_cache: dict[tuple[SCTID, SCTID], bool] = {}

    def _get(self, *args, **kwargs) -> requests.Response:
        """\
Wrapper for requests.get that prepends known url and
raises an exception on non-200 responses.
"""
        # Check for the timeout
        if PROFILING and STOP_AFTER_SECONDS is not None:
            elapsed = datetime.datetime.now() - self.__start_time
            if elapsed.total_seconds() > STOP_AFTER_SECONDS:
                raise ProfileMark("Time limit exceeded")

        # Include the known url
        if "url" not in kwargs:
            args = (self.url + args[0], *args[1:])
        else:
            kwargs["url"] = self.url + kwargs["url"]

        kwargs["headers"] = kwargs.get("headers", {})
        kwargs["headers"]["Accept"] = "application/json"

        response = requests.get(*args, **kwargs)
        if not response.ok:
            raise SnowstormRequestError.from_response(response)

        return response

    def _get_collect(self, *args, **kwargs) -> list:
        """\
Wrapper for requests.get that collects all items from a paginated response.
"""
        total = None
        offset = 0
        step = self.PAGINATION_STEP
        collected_items = []

        while total is None or offset < total:
            kwargs["params"] = kwargs.get("params", {})
            kwargs["params"]["offset"] = offset
            kwargs["params"]["limit"] = step

            response = self._get(*args, **kwargs)

            collected_items.extend(response.json()["items"])
            total = response.json()["total"]
            offset += step

        return collected_items

    def get_version(self) -> str:
        response = self._get("version")
        return response.json()["version"]

    def get_main_branch_path(self) -> BranchPath:
        # Get codesystems and look for a target
        response = self._get("codesystems")

        for codesystem in response.json()["items"]:
            if codesystem["shortName"] == self.TARGET_CODESYSTEM:
                # TODO: double-check by module contents
                return BranchPath(codesystem["branchPath"])

        raise SnowstormAPIError(
            f"Target codesystem {self.TARGET_CODESYSTEM} is not present"
        )

    def get_concept(self, sctid: SCTID) -> Concept:
        """\
Get full concept information.
"""
        if sctid in self.__concepts_cache:
            return self.__concepts_cache[sctid]
        response = self._get(f"browser/{self.branch_path}/concepts/{sctid}")
        concept = Concept.from_json(response.json())
        self.__concepts_cache[sctid] = concept
        return concept

    def get_branch_info(self) -> dict:
        response = self._get(f"branches/{self.branch_path}")
        return response.json()

    def get_attribute_suggestions(
        self, parent_ids: Iterable[SCTID]
    ) -> dict[SCTID, AttributeConstraints]:
        params = {
            "parentIds": [*parent_ids],
            "proximalPrimitiveModeling": True,  # Maybe?
            # Filter post-coordination for now
            "contentType": "ALL",
        }

        response = self._get(
            url="mrcm/" + self.branch_path + "/domain-attributes",
            params=params,
        )

        return {
            SCTID(attr["id"]): AttributeConstraints.from_json(attr)
            for attr in response.json()["items"]
            if SCTID(attr["id"]) != IS_A  # Exclude hierarchy
        }

    def get_mrcm_domain_reference_set_entries(
        self,
    ) -> list[dict]:
        collected_items = self._get_collect(
            url=f"{self.branch_path}/members",
            params={
                "referenceSet": MRCM_DOMAIN_REFERENCE_SET_ECL,
                "active": True,
            },
        )
        return collected_items

    @staticmethod
    def _range_constraint_to_parents(
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
        TERM_ = r"\|(?P<term>.+?) \([a-z]+(?: [a-z]+)*\)\|"
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
                logging.error(f"{rc} is not a simple disjunction of SCTIDs!")
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
        logging.warning(f"No range constraint found for {attribute}")
        return {}

    def get_concept_children(
        self,
        parent: SCTID,
        require_property: Mapping[str, JsonPrimitive] | None = None,
    ) -> dict[SCTID, SCTDescription]:
        response = self._get(
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

    def is_concept_descendant_of(
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

        response = self._get(
            url=f"{self.branch_path}/concepts/",
            params={
                "ecl": f"<{parent}",  # Is a subtype of
                "conceptIds": [child],  # limit to known child
            },
        )

        out = bool(response.json()["total"])  # Should be 1 or 0

        self.__subsumptions_cache[(child, parent)] = out  # Cache the result
        return out

    def filter_bad_descendants(
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
        for bad_batch in itertools.batched(
            bad_parents, self.MAX_BAD_PARENT_QUERY
        ):
            expression = " OR ".join(f"<{b_p}" for b_p in sorted(bad_batch))
            actual_children = self._get_collect(
                url=f"{self.branch_path}/concepts/",
                params={
                    "ecl": expression,
                    "returnIdOnly": True,
                    "conceptIds": sorted(out),  # limit to known children
                },
            )
            out -= set(SCTID(c) for c in actual_children)
            if not out:
                break

        return out

    def is_attr_val_descendant_of(
        self, child: AttributeRelationship, parent: AttributeRelationship
    ) -> bool:
        """\
Check if the child attribute-value pair is a subtype of the parent.
"""
        attribute_is_child = self.is_concept_descendant_of(
            child.attribute, parent.attribute
        )
        value_is_child = self.is_concept_descendant_of(
            child.value, parent.value
        )
        return attribute_is_child and value_is_child

    def remove_redundant_ancestors(self, portrait: SemanticPortrait) -> None:
        """\
Remove ancestors that are descendants of other ancestors.
"""
        redundant_ancestors = set()
        for ancestor in portrait.ancestor_anchors:
            if any(
                self.is_concept_descendant_of(
                    other, ancestor, self_is_parent=False
                )
                for other in portrait.ancestor_anchors
            ):
                redundant_ancestors.add(ancestor)
        portrait.ancestor_anchors -= redundant_ancestors

    def get_concept_ppp(self, concept: SCTID) -> set[SCTID]:
        """\
Get a concept's Proximal Primitive Parents
"""
        response = self._get(
            f"{self.branch_path}/concepts/{concept}/normal-form",
        )

        focus_concepts_string = response.json()["expression"].split(" : ")[0]
        concepts = map(SCTID, focus_concepts_string.split(" + "))

        out = set()
        for concept in concepts:
            concept_info = self.get_concept(concept)
            if concept_info.defined:
                # Recurse for defined concepts
                out |= self.get_concept_ppp(concept)
            else:
                out.add(concept)

        return out

    def check_inferred_subsumption(
        self, parent_predicate: Concept, portrait: SemanticPortrait
    ) -> bool:
        """\
Check if the particular portrait can be a subtype of a parent concept.

Note that subsumption is checked for concepts regardless of definition status;
Primitive concepts will report subsumption as True, but it needs to be confirmed
manually/with LLM.
"""
        logging.debug(
            "Checking subsumption for",
            portrait.source_term,
            "under",
            parent_predicate.pt,
        )

        # To be considered eligible as a descendant, all the predicate's PPP
        # must be ancestors of at least one anchor
        unmatched_predicate_ppp: set[SCTID] = self.get_concept_ppp(
            parent_predicate.sctid
        )
        for anchor in portrait.ancestor_anchors:
            matched_ppp: set[SCTID] = set()
            for ppp in unmatched_predicate_ppp:
                if self.is_concept_descendant_of(anchor, ppp):
                    matched_ppp.add(ppp)
            unmatched_predicate_ppp -= matched_ppp

        if unmatched_predicate_ppp:
            logging.debug(
                f"Does not satisfy {len(unmatched_predicate_ppp)} "
                + "PPP constraints"
            )
            return False

        # For now, we do not worry about the groups; we may have to once
        # we allow multiple of a same attribute
        unmatched_concept_relationships: set[AttributeRelationship] = set()
        for group in parent_predicate.groups:
            unmatched_concept_relationships |= group.relationships
        unmatched_concept_relationships |= parent_predicate.ungrouped

        for av in portrait.attributes.items():
            p_rel = AttributeRelationship(*av)
            matched_attr: set[AttributeRelationship] = set()
            for c_rel in unmatched_concept_relationships:
                if self.is_attr_val_descendant_of(p_rel, c_rel):
                    matched_attr.add(c_rel)
            unmatched_concept_relationships -= matched_attr
            if not unmatched_concept_relationships:
                # Escape early if all relationships are matched
                break

        if unmatched := len(unmatched_concept_relationships):
            logging.debug(f"Does not satisfy {unmatched} attribute constraints")
            return False

        logging.debug(
            "All constraints are satisfied",
            "but concept is primitive!" * (not parent_predicate.defined),
        )
        return True


# Main logic host
class Bouzyges:
    """\
Main logic host for the Bouzyges system.
"""

    def __init__(
        self,
        snowstorm: SnowstormAPI,
        prompter: Prompter,
        terms: Iterable[str],
        contexts: Iterable[str] | None = None,
    ):
        self.snowstorm = snowstorm
        self.prompter = prompter
        self.portraits = {}
        if contexts is None:
            self.portraits = {term: SemanticPortrait(term) for term in terms}
        else:
            self.portraits = {
                term: SemanticPortrait(term, context)
                for term, context in zip(terms, contexts)
            }

        logging.info("Snowstorm Version:" + snowstorm.get_version())
        logging.info("Using branch path:", snowstorm.branch_path)

        # Load MRCM entries
        logging.info("MRCM Domain Reference Set entries:")
        domain_entries = snowstorm.get_mrcm_domain_reference_set_entries()
        logging.info("Total entries:", len(domain_entries))
        self.mrcm_entries = [
            MRCMDomainRefsetEntry.from_json(entry)
            for entry in domain_entries
            if SCTID(entry["referencedComponent"]["conceptId"])
            in WHITELISTED_SUPERTYPES
        ]

        for entry in self.mrcm_entries:
            logging.info(" -", entry.term + ":")
            logging.info("    -", entry.domain_constraint)
            logging.info("    -", entry.guide_link)

        # Initialize supertypes
        self.initialize_supertypes()

    def initialize_supertypes(self):
        """\
Initialize supertypes for all terms to start building portraits.
"""
        for source_term, portrait in self.portraits.items():
            if portrait.ancestor_anchors:
                raise BouzygesError(
                    "Should not happen: ancestor anchors are set, "
                    "and yet initialize_supertypes is called"
                )

            supertypes_decode = {
                entry.term: entry.sctid for entry in self.mrcm_entries
            }
            supertype_term = prompter.prompt_supertype(
                term=portrait.source_term,
                options=supertypes_decode,
                allow_escape=False,
                term_context="; ".join(portrait.context)
                if portrait.context
                else None,
            )
            match supertype_term:
                case SCTDescription(answer_term):
                    supertype = supertypes_decode[answer_term]
                    logging.info("Assuming", source_term, "is", answer_term)
                    portrait.ancestor_anchors.add(supertype)
                case _:
                    raise BouzygesError(
                        "Should not happen: null-like response from prompter; "
                        "did the Prompter inject the escape hatch?"
                    )

    def populate_attribute_candidates(self) -> None:
        for term, portrait in self.portraits.items():
            attributes = self.snowstorm.get_attribute_suggestions(
                portrait.ancestor_anchors
            )

            # Remove previously rejected attributes
            for attribute in portrait.rejected_attributes:
                attributes.pop(attribute, None)

            logging.debug("Possible attributes for:", term)
            for sctid, attribute in attributes.items():
                logging.debug(" - ", sctid, attribute.pt)

            # Confirm the attributes
            for attribute in attributes.values():
                accept = self.prompter.prompt_attr_presence(
                    term=portrait.source_term,
                    attribute=attribute.pt,
                    term_context="; ".join(portrait.context)
                    if portrait.context
                    else None,
                )
                logging.info(
                    attribute.sctid,
                    attribute.pt + ":",
                    "Present" if accept else "Not present",
                )

                if accept:
                    portrait.unchecked_attributes.add(attribute.sctid)

            # Remember the constraints
            for sctid, attribute in attributes.items():
                if sctid not in portrait.unchecked_attributes:
                    continue
                portrait.relevant_constraints[sctid] = attribute

    def populate_unchecked_attributes(self) -> None:
        for portrait in self.portraits.values():
            rejected = set()
            for attribute in portrait.unchecked_attributes:
                logging.debug("Attribute:", attribute)
                # Get possible attribute values
                values_options = self.snowstorm.get_attribute_values(
                    portrait, attribute
                )
                logging.debug("Values:", values_options)

                if not values_options:
                    # No valid values for this attribute and parent combination
                    rejected.add(attribute)
                    continue
                else:
                    logging.info("Possible values for:", attribute)
                    for value in values_options:
                        logging.info(" -", value)

                # Prompt for the value
                value_term = self.prompter.prompt_attr_value(
                    term=portrait.source_term,
                    attribute=portrait.relevant_constraints[attribute].pt,
                    options=values_options.values(),
                    term_context="; ".join(portrait.context)
                    if portrait.context
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
                        portrait.attributes[attribute] = sctid
                    case EscapeHatch.WORD:
                        # Choosing no attribute on initial prompt
                        # means rejection
                        rejected.add(attribute)

            portrait.rejected_attributes |= rejected
            # All are seen by now
            portrait.unchecked_attributes = set()

    def update_existing_attr_values(self) -> None:
        """\
Update existing attribute values with the most precise descendant for all terms.
"""
        for source_term, portrait in self.portraits.items():
            new_attributes = {}
            for attribute, value in portrait.attributes.items():
                new_attributes[attribute] = value
                while True:
                    # Get children of the current value
                    children = self.snowstorm.get_concept_children(
                        new_attributes[attribute]
                    )
                    if not children:
                        # Leaf node
                        break

                    descriptions = {v: k for k, v in children.items()}

                    # Prompt for the most precise value
                    value_term = self.prompter.prompt_attr_value(
                        term=source_term,
                        attribute=portrait.relevant_constraints[attribute].pt,
                        options=descriptions,
                        term_context="; ".join(portrait.context)
                        if portrait.context
                        else None,
                        allow_escape=True,
                    )

                    if isinstance(value_term, EscapeHatch):
                        # None of the children are correct
                        break

                    new_attributes[attribute] = descriptions[value_term]

            portrait.attributes.update(new_attributes)

    def update_anchors(self, source_term: str) -> bool:
        portrait = self.portraits[source_term]
        i = 0
        ancestors_changed = True
        while ancestors_changed:
            ancestors_changed = self.__update_anchor(portrait)
            i += ancestors_changed
        logging.info(f"Updated {source_term} anchors in {i} iterations.")
        return i > 1

    def __update_anchor(
        self,
        portrait: SemanticPortrait,
    ) -> bool:
        """\
Update the ancestor anchors for one term to more precise children. Performs a
single iteration. Return True if the parent anchors have changed, False
otherwise.
"""
        new_anchors = set()
        for anchor in portrait.ancestor_anchors:
            # Get all immediate descendants
            children = self.snowstorm.get_concept_children(anchor)
            logging.debug(f"Filtering {len(children)} children of {anchor}")

            # Remove verbatim known ancestors
            for known_ancestor in portrait.ancestor_anchors:
                children.pop(known_ancestor, None)

            # Filter previously rejected ancestors including meta-ancestors
            remaining = self.snowstorm.filter_bad_descendants(
                children=children, bad_parents=portrait.rejected_supertypes
            )

            # Save the rejected children as rejected ancestors
            portrait.rejected_supertypes.update(
                child for child in children if child not in remaining
            )

            for child in set(children) - remaining:
                del children[child]

            logging.debug(f"Filtered to {len(children)}")

            # Iterate over descendants and ask LLM/Snowstorm if to include them
            # to the new anchors
            for child_id, child_term in children.items():
                child_concept = self.snowstorm.get_concept(child_id)

                supertype: bool = self.snowstorm.check_inferred_subsumption(
                    child_concept, portrait
                )

                if not supertype:
                    # This will prevent expensive re-queries on descendants
                    portrait.rejected_supertypes.add(child_id)
                    continue

                # Primitive concepts must be confirmed by the LLM
                primitive = not child_concept.defined
                term_context = (
                    "; ".join(portrait.context) if portrait.context else None
                )
                if primitive and not self.prompter.prompt_subsumption(
                    term=portrait.source_term,
                    prospective_supertype=child_term,
                    term_context=term_context,
                ):
                    portrait.rejected_supertypes.add(child_id)
                    continue

                new_anchors.add(child_id)

        if not new_anchors:
            logging.debug("No new ancestors found")
            return False

        # Update the anchor set with the new one
        logging.debug("New ancestors:", new_anchors - portrait.ancestor_anchors)
        portrait.ancestor_anchors |= new_anchors
        return True

    def run(self):
        start_time = datetime.datetime.now()
        logging.info("Started at:", start_time)

        """Main routine"""
        self.populate_attribute_candidates()
        self.populate_unchecked_attributes()
        self.update_existing_attr_values()
        logging.info("Attributes:")
        for term, portrait in self.portraits.items():
            logging.info(" -", term, "attributes:")
            for attribute, value in portrait.attributes.items():
                logging.info("   -", attribute, value)

        for term, portrait in self.portraits.items():
            changes_made = updated = True

            while changes_made:
                cycles = 0
                while updated:
                    updated = self.update_anchors(term)
                    cycles += updated
                changes_made = bool(cycles)

            self.snowstorm.remove_redundant_ancestors(portrait)

        # Print resulting supertypes
        for term, portrait in self.portraits.items():
            print(term, "supertypes:")
            for anchor in portrait.ancestor_anchors:
                ancestor = self.snowstorm.get_concept(anchor)
                print(" -", ancestor.sctid, ancestor.pt)

        logging.info("Started at:", start_time)
        logging.info(
            "Time taken (s):",
            (datetime.datetime.now() - start_time).total_seconds(),
        )
        self.prompter.report_usage()


if __name__ == "__main__":
    # Load environment variables for API access
    if dotenv is not None:
        logging.info("Loading environment variables from .env")
        dotenv.load_dotenv()

    snowstorm_endpoint = os.getenv("SNOWSTORM_ENDPOINT")
    if not snowstorm_endpoint:
        raise ValueError("SNOWSTORM_ENDPOINT environment variable is not set")
    snowstorm = SnowstormAPI(URL(snowstorm_endpoint))

    # If API key exists in the environment, use it
    prompter: Prompter
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if api_key is not None and api_endpoint is not None:
        prompter = OpenAIAzurePrompter(
            prompt_format=OpenAIPromptFormat(),
            api_key=api_key,
            azure_endpoint=api_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        if not prompter.ping():
            logging.warn("Azure API is not available, using human prompter.")
            prompter = HumanPrompter(prompt_format=VerbosePromptFormat())
    else:
        logging.warn("No Azure API key found, using human prompter.")
        prompter = HumanPrompter(prompt_format=VerbosePromptFormat())

    # Test
    bouzyges = Bouzyges(
        snowstorm=snowstorm,
        prompter=prompter,
        terms=["Pyogenic liver abscess"],
    )

    if PROFILING:
        with cProfile.Profile() as prof:
            try:
                bouzyges.run()
            except ProfileMark:
                pass
        stats = pstats.Stats(prof)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats("stats.prof")
    else:
        bouzyges.run()
