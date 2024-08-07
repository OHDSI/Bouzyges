import dotenv
import enum
import json
import openai
import os
import re
import requests
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from frozendict import frozendict
from typing import Iterable, Mapping


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

        return cls(
            sctid=SCTID(json_data["conceptId"]),
            pt=SCTDescription(json_data["pt"]["term"]),
            fsn=SCTDescription(json_data["fsn"]["term"]),
            groups=frozenset(
                AttributeGroup(frozenset(relationships))
                for relationships in groups.values()
            ),
            ungrouped=frozenset(ungrouped),
            defined=json_data["definitionStatus"] == "FULLY_DEFINED",
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

    class State(enum.Enum):
        # Completed state
        COMPLETED = 0  # Ready for use
        # Incomplete states
        MISSING_SUPERTYPES = 1  # No supertypes selected
        MISSING_ATTRIBUTES = 2  # No attributes selected
        MISSING_ATTRIBUTE_VALUES = 3  # Not all attributes have values
        # Interim states
        ATTRIBUTES_FILLED = 4  # All attributes have values or are rejected
        PRIMITIVE_EXHAUSTED = 5  # Proximal primitive anchors are set
        CLASSIFIED = 6  # Defined supertypes are set through classification

    def __init__(self, term: str, context: str | None = None) -> None:
        self.source_term: str = term
        self.context: Iterable[str] | None = context
        self.ancestor_anchors: set[SCTID] = set()

        self.unchecked_attributes: set[SCTID] = set()
        self.attributes: dict[SCTID, SCTID] = {}
        self.rejected_attributes: set[SCTID] = set()
        self.rejected_supertypes: set[SCTID] = set()

        self.state = self.State.MISSING_SUPERTYPES
        self.relevant_constraints: dict[SCTID, AttributeConstraints] = {}

    def check_state(self) -> None:
        if not self.ancestor_anchors:
            self.state = self.State.MISSING_SUPERTYPES
        elif not any(self.rejected_attributes) and not any(self.attributes):
            self.state = self.State.MISSING_ATTRIBUTES
        elif self.unchecked_attributes:
            self.state = self.State.MISSING_ATTRIBUTE_VALUES
        else:
            # Portrait does not know if it or it's
            # attributes have unresolved subtypes; Nor if
            # classification is complete.
            # It will report self as completed
            self.state = self.State.COMPLETED


## Exceptions
class SnowstormAPIError(Exception):
    """Raised when the Snowstorm API returns a bad response."""


class SnowstormRequestError(SnowstormAPIError):
    """Raised when the Snowstorm API returns a non-200 response"""

    def __init__(self, text, response, *_):
        super().__init__(text)
        self.response = response

    @classmethod
    def from_response(cls, response):
        print(json.dumps(response.json(), indent=2))
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

    prompt_message: str
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

    def get(self, model: str, prompt: Prompt) -> str | None:
        """\
Get the answer from the cache for specified model.
"""
        query = f"""
            SELECT response
            FROM {self.table_name}
            WHERE
                model = ? AND
                prompt = ? AND
                api_options
        """
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
        cursor = self.connection.cursor()
        cursor.execute(
            query,
            (model, prompt.prompt_message, api_options),
        )
        if answer := cursor.fetchone():
            return answer[0]
        return None

    def remember(self, model: str, prompt: Prompt, response: str) -> None:
        """\
Remember the answer for the prompt for the specified model.
"""
        query = f"""
            INSERT INTO {self.table_name} (model, prompt, response, api_options)
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
        cursor.execute(
            query,
            (model, prompt.prompt_message, response, api_options),
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

Contains no API options, intended for human prompters.
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

    # Following methods are abstract and represent common queries to the model
    @abstractmethod
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

    @abstractmethod
    def prompt_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> bool:
        """\
Prompt the model to decide if an attribute is present in a term.
"""

    @abstractmethod
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
        """\
Prompt the model to choose the value of an attribute in a term.
"""

    @abstractmethod
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

    @abstractmethod
    def ping(self) -> bool:
        """\
Check if the API is available.
"""

    def cache_remember(self, prompt: Prompt, answer: str) -> None:
        if self.cache:
            self.cache.remember(self._model_id, prompt, answer)


class HumanPrompter(Prompter):
    """\
A test prompter that interacts with a human to get answers.
"""

    _model_id = "human"

    def prompt_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> SCTDescription | EscapeHatch:
        # Construct the prompt
        prompt: Prompt = self.prompt_format.form_supertype(
            term, options, allow_escape, term_context, options_context
        )

        # Try the cache
        if self.cache:
            if cached_answer := self.cache.get(self._model_id, prompt):
                answer = self.unwrap_class_answer(
                    cached_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
                print(f"From cache: {answer} is a supertype of {term}")
                return answer

        # Get the answer
        return self._prompt_class_answer(allow_escape, options, prompt)

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
                print(
                    f"From cache: The attribute '{attribute}' is "
                    f"{'present' if answer else 'absent'} in '{term}'"
                )
                return answer

        return self._prompt_bool_answer(prompt)

    def _prompt_bool_answer(self, prompt):
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

        if self.cache:
            if cached_answer := self.cache.get(self._model_id, prompt):
                answer = self.unwrap_class_answer(
                    cached_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
                print(
                    f"From cache: The value of the attribute '{attribute}' in "
                    f"'{term}' is '{answer}'"
                )
                return answer

        return self._prompt_class_answer(allow_escape, options, prompt)

    def prompt_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> bool:
        prompt: Prompt = self.prompt_format.form_subsumption(
            term, prospective_supertype, term_context, supertype_context
        )

        if self.cache:
            if cached_answer := self.cache.get(self._model_id, prompt):
                answer = self.unwrap_bool_answer(cached_answer)
                print(
                    f"From cache: The term '{term}' is",
                    "a subtype" if answer else "not a subtype",
                    f"of '{prospective_supertype}'",
                )
                return answer

        return self._prompt_bool_answer(prompt)

    def ping(self) -> bool:
        input("Hey, are you here? (Press Enter) ")
        print("Good human.")
        return True


class OpenAIAzurePrompter(Prompter):
    """\
A prompter that interfaces with the OpenAI API using Azure.
"""

    DEFAULT_MODEL = "gpt-35-turbo"
    DEFAULT_VERSION = "2023-07-01-preview"

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

        print("Initializing the Azure API client...")
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version or self.DEFAULT_VERSION,
        )
        self._endpoint = azure_endpoint
        self._model_id = model or self.DEFAULT_MODEL

    def ping(self) -> bool:
        print("Pinging the Azure API...")
        # Ping by retrieving the list of models
        headers = {"api-key": self._client.api_key}
        try:
            models = self._client.models.list(extra_headers=headers, timeout=5)
        except openai.APITimeoutError:
            print("Connection timed out")
            return False
        response: dict = models.model_dump()
        success = response.get("data", []) != []
        if success:
            print("API is available")
            if self._model_id not in [obj["id"] for obj in response["data"]]:
                print(
                    f"But '{self._model_id}' is not present in the API response!"
                )
                return False
            return True

        print("API is not available")
        return False

    def prompt_supertype(
        self,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> SCTDescription | EscapeHatch:
        _ = term, options, allow_escape, term_context, options_context
        raise NotImplementedError

    def prompt_attr_presence(
        self,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> bool:
        _ = term, attribute, term_context, attribute_context
        raise NotImplementedError

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
        _ = (
            term,
            attribute,
            options,
            term_context,
            attribute_context,
            options_context,
            allow_escape,
        )
        raise NotImplementedError

    def prompt_subsumption(
        self,
        term: str,
        prospective_supertype: SCTDescription,
        term_context: str | None = None,
        supertype_context: str | None = None,
    ) -> bool:
        _ = term, prospective_supertype, term_context, supertype_context
        raise NotImplementedError


class SnowstormAPI:
    TARGET_CODESYSTEM = "SNOMEDCT"
    CONTENT_TYPE_PREFERENCE = "NEW_PRECOORDINATED", "PRECOORDINATED", "ALL"
    PAGINATION_STEP = 100

    def __init__(self, url: URL):
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

    def _get_collect(self, *args, **kwargs) -> list[dict]:
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

    @staticmethod
    def print_attribute_model_hierarchy(amh: dict) -> None:
        def print_node(node, indent):
            print("  " * indent, f"{node['conceptId']}:", node["pt"]["term"])
            for child in node.get("children", ()):
                print_node(child, indent + 1)

        print_node(amh, 0)

    def get_mrcm_domain_reference_set_entries(
        self,
    ) -> list[dict]:
        collected_items = self._get_collect(
            url=f"/{self.branch_path}/members",
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
                print(f"{rc} is not a simple disjunction of SCTIDs!")
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
        print(f"No range constraint found for {attribute}")
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
        if self_is_parent and child == parent:
            return True

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
            for other in portrait.ancestor_anchors:
                if ancestor == other:
                    continue
                if self.is_concept_descendant_of(
                    other, ancestor, self_is_parent=False
                ):
                    redundant_ancestors.add(ancestor)
                break
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
        self, parent_predicate: SCTID, portrait: SemanticPortrait
    ) -> bool:
        """\
Check if the particular portrait can be a subtype of a parent concept.
"""
        parent_concept = self.get_concept(parent_predicate)
        print(
            "Checking subsumption for",
            portrait.source_term,
            "under",
            parent_concept.pt,
        )
        if not parent_concept.defined:
            # Subsumption is not possible
            print("Parent is not defined")
            return False

        # To be considered eligible as a descendant, all the predicate's PPP
        # must be ancestors of at least one anchor
        unmatched_predicate_ppp: set[SCTID] = self.get_concept_ppp(
            parent_predicate
        )
        for anchor in portrait.ancestor_anchors:
            matched_ppp: set[SCTID] = set()
            for ppp in unmatched_predicate_ppp:
                if self.is_concept_descendant_of(anchor, ppp):
                    matched_ppp.add(ppp)
            unmatched_predicate_ppp -= matched_ppp

        if unmatched_predicate_ppp:
            print(
                f"Does not satisfy {len(unmatched_predicate_ppp)} "
                + "PPP constraints"
            )
            return False

        # For now, we do not worry about the groups; we may have to once
        # we allow multiple of a same attribute
        unmatched_concept_relationships: set[AttributeRelationship] = set()
        for group in parent_concept.groups:
            unmatched_concept_relationships |= group.relationships
        unmatched_concept_relationships |= parent_concept.ungrouped

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
            print(f"Does not satisfy {unmatched} attribute constraints")
            return False

        print("All constraints are satisfied")
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

        print("Snowstorm Version:" + snowstorm.get_version())
        print("Using branch path:", snowstorm.branch_path)

        # Load MRCM entries
        print("MRCM Domain Reference Set entries:")
        domain_entries = snowstorm.get_mrcm_domain_reference_set_entries()
        print("Total entries:", len(domain_entries))
        self.mrcm_entries = [
            MRCMDomainRefsetEntry.from_json(entry)
            for entry in domain_entries
            if SCTID(entry["referencedComponent"]["conceptId"])
            in WHITELISTED_SUPERTYPES
        ]

        for entry in self.mrcm_entries:
            print(" -", entry.term + ":")
            print("    -", entry.domain_constraint)
            print("    -", entry.guide_link)
            print()

        # Initialize supertypes
        self.initialize_supertypes()

    def initialize_supertypes(self):
        """\
Initialize supertypes for all terms to start building portraits.
"""
        for source_term, portrait in self.portraits.items():
            if portrait.ancestor_anchors:
                raise ValueError(
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
                    print("Assuming", source_term, "is", answer_term)
                    portrait.ancestor_anchors.add(supertype)
                case _:
                    raise ValueError(
                        "Should not happen: "
                        "did the prompter inject the escape hatch?"
                    )

    def populate_attribute_candidates(self) -> None:
        for term, portrait in self.portraits.items():
            attributes = self.snowstorm.get_attribute_suggestions(
                portrait.ancestor_anchors
            )

            # Remove previously rejected attributes
            for attribute in portrait.rejected_attributes:
                attributes.pop(attribute, None)

            print("Possible attributes for:", term)
            for sctid, attribute in attributes.items():
                print(" - ", sctid, attribute.pt)

            # Confirm the attributes
            for attribute in attributes.values():
                accept = self.prompter.prompt_attr_presence(
                    term=portrait.source_term,
                    attribute=attribute.pt,
                    term_context="; ".join(portrait.context)
                    if portrait.context
                    else None,
                )
                print(
                    " -",
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
                print("Attribute:", attribute)
                # Get possible attribute values
                values_options = self.snowstorm.get_attribute_values(
                    portrait, attribute
                )
                print("Values:", values_options)

                if not values_options:
                    # No valid values for this attribute and parent combination
                    rejected.add(attribute)
                    continue
                else:
                    print("Possible values for:", attribute)
                    for value in values_options:
                        print(" -", value)

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

    def update_anchors(self, source_term: str) -> bool:
        portrait = self.portraits[source_term]
        i = 0
        ancestors_changed = True
        while ancestors_changed:
            ancestors_changed = self.__update_anchor(portrait)
            i += ancestors_changed
        print(f"Updated {source_term} anchors in {i} iterations.")
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
            print(f"Filtering {len(children)} children of {anchor}")

            # Filter previously rejected ancestors including meta-ancestors
            removed = set()
            for child in children:
                if any(
                    self.snowstorm.is_concept_descendant_of(child, bad)
                    for bad in portrait.rejected_supertypes
                ):
                    removed.add(child)
                    portrait.rejected_supertypes.add(child)
                    break

            children = {
                k: v
                for k, v in children.items()
                # Do not revisit denied children or known ancestors
                if k not in removed | portrait.ancestor_anchors
            }

            print(f"Filtered to {len(children)}")

            # Iterate over descendants and ask LLM/Snowstorm if to include them
            # to the new anchors
            for child_id, child_term in children.items():
                if self.snowstorm.get_concept(child_id).defined:
                    include = self.snowstorm.check_inferred_subsumption(
                        child_id, portrait
                    )
                else:
                    include = self.prompter.prompt_subsumption(
                        term=portrait.source_term,
                        prospective_supertype=child_term,
                        term_context=portrait.context,
                    )

                if include:
                    new_anchors.add(child_id)
                else:
                    # This will prevent expensive re-queries on descendants
                    portrait.rejected_supertypes.add(child_id)

        if not new_anchors:
            print("No new ancestors found")
            return False

        # Update the anchor set with the new one
        print("New ancestors:", new_anchors - portrait.ancestor_anchors)
        portrait.ancestor_anchors |= new_anchors
        return True


if __name__ == "__main__":
    # Load environment variables for API access
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
            prompt_format=VerbosePromptFormat(),
            api_key=api_key,
            azure_endpoint=api_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        if not prompter.ping():
            print("Azure API is not available, using human prompter.")
            prompter = HumanPrompter(prompt_format=VerbosePromptFormat())
    else:
        print("No Azure API key found, using human prompter.")
        prompter = HumanPrompter(prompt_format=VerbosePromptFormat())

    # Test
    bouzyges = Bouzyges(
        snowstorm=snowstorm,
        prompter=prompter,
        terms=["Pyogenic liver abscess"],
    )

    bouzyges.populate_attribute_candidates()
    bouzyges.populate_unchecked_attributes()
    print("Attributes:")
    for term, portrait in bouzyges.portraits.items():
        print(" -", term, "attributes:")
        for attribute, value in portrait.attributes.items():
            print("   -", attribute, value)

    bouzyges.update_existing_attr_values()
    for term, portrait in bouzyges.portraits.items():
        changes_made = updated = True

        while changes_made:
            cycles = 0
            while updated:
                updated = bouzyges.update_anchors(term)
                cycles += updated
            changes_made = bool(cycles)

            bouzyges.snowstorm.remove_redundant_ancestors(portrait)

    # Print resulting supertypes
    for term, portrait in bouzyges.portraits.items():
        print(term, "supertypes:")
        for anchor in portrait.ancestor_anchors:
            ancestor = bouzyges.snowstorm.get_concept(anchor)
            print(" -", ancestor.sctid, ancestor.pt)
