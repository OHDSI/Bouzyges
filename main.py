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
from typing import Iterable


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

    group_id: SCTID
    relationships: frozenset[AttributeRelationship]


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
                # Use pre-coordination: stricter
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


@dataclass(frozen=True, slots=True)
class AttributeConstraints:
    """\
Represents information about an attribute obtained from a known set of parent 
concepts.
"""

    sctid: SCTID
    pt: SCTDescription
    attributeDomain: Iterable[AttributeDomain]
    # May actually not be required
    # because /mrcm/{branch}/attribute-values/ exists
    attributeRange: Iterable[AttributeRange]

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            sctid=SCTID(json_data["conceptId"]),
            pt=SCTDescription(json_data["pt"]["term"]),
            attributeDomain=[
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
            attributeRange=[
                AttributeRange(
                    sctid=SCTID(json_data["conceptId"]),
                    pt=SCTDescription(json_data["pt"]["term"]),
                    range_constraint=ECLExpression(ar["rangeConstraint"]),
                    attribute_rule=ECLExpression(ar["attributeRule"]),
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

    def __init__(self, term: str, context: str | None = None) -> None:
        self.source_term: str = term
        self.context: Iterable[str] | None = context
        self.ancestor_anchors: set[SCTID] = set()

        self.unchecked_attributes: set[SCTID] = set()
        self.attributes: dict[SCTID, SCTID] = {}
        self.rejected_attributes: set[SCTID] = set()

        self.state = self.State.MISSING_SUPERTYPES

    def update_state(self) -> None:
        if not self.ancestor_anchors:
            self.state = self.State.MISSING_SUPERTYPES
        elif not any(self.rejected_attributes) and not any(self.attributes):
            self.state = self.State.MISSING_ATTRIBUTES
        elif self.unchecked_attributes:
            self.state = self.State.MISSING_ATTRIBUTE_VALUES
        else:
            # Portrait does not know if it or it's
            # attributes have unresolved subtypes;
            # It will report as completed
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
    api_options: frozendict[str, str | int | float | bool] | None = None


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
        return cursor.fetchone()

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

    @classmethod
    def form_supertype(
        cls,
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
        _ = term, options, allow_escape, term_context, options_context
        raise NotImplementedError

    @classmethod
    def form_attr_presence(
        cls,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> Prompt:
        """\
Format a prompt for the LLM agent to decide if an attribute is present in a
term.
"""
        _ = term, attribute, term_context, attribute_context
        raise NotImplementedError

    @classmethod
    def form_attr_value(
        cls,
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


class DefaultPromptFormat(PromptFormat):
    """\
Default simple verbose prompt format for the LLM agent.

Contains no API options, intended for human prompters.
    """

    @classmethod
    def _form_shared_header(cls, allow_escape) -> str:
        prompt = ""
        prompt += "You are " + cls.ROLE + ".\n\n"
        prompt += "Your assignment is " + cls.TASK + ".\n\n"
        prompt += "Please note that " + cls.REQUIREMENTS + ".\n\n"
        prompt += "Your exact instructions are:\n\n"
        prompt += cls.INSTRUCTIONS + (allow_escape * cls.ESCAPE_INSTRUCTIONS)
        prompt += "\n\n"
        return prompt

    @classmethod
    def form_supertype(
        cls,
        term: str,
        options: Iterable[SCTDescription],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
    ) -> Prompt:
        prompt = cls._form_shared_header(allow_escape)

        prompt += (
            f"Given the term '{term}', what is the closest supertype or exact "
            "equivalent of it's meaning from the following options?"
        )
        if term_context:
            prompt += " Following information is provided about the term: "
            prompt += term_context

        prompt += "\n\nOptions, in no particular order:\n"
        for option in options:
            prompt += f" - {cls.wrap_term(option)}"
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

    @classmethod
    def form_attr_presence(
        cls,
        term: str,
        attribute: SCTDescription,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> Prompt:
        prompt = cls._form_shared_header(allow_escape=False)
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

    @classmethod
    def form_attr_value(
        cls,
        term: str,
        attribute: SCTDescription,
        options: Iterable[SCTDescription],
        term_context: str | None = None,
        attribute_context: str | None = None,
        options_context: dict[SCTDescription, str] | None = None,
        allow_escape: bool = True,
    ) -> Prompt:
        prompt = cls._form_shared_header(allow_escape=allow_escape)
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
        for option in options:
            prompt += f" - {cls.wrap_term(option)}"
            if options_context and option in options_context:
                prompt += f": {options_context[option]}"
            prompt += "\n"
        # Remind of the escape hatch, just in case
        prompt += f" - {EscapeHatch.WORD}: " "None of the above\n"
        return Prompt(prompt, frozenset(options), EscapeHatch.WORD)


class OpenAIPromptFormat(PromptFormat):
    """\
Default prompt format for the OpenAI API.
"""


## Logic prompter classes
class Prompter(ABC):
    """\
Interfaces prompts to the LLM agent and parses answers.
"""

    def __init__(
        self,
        *args,
        prompt_format: PromptFormat,
        use_cache: bool = True,
        **kwargs,
    ):
        _ = args, kwargs
        self.prompt_format = prompt_format
        self.cache: sqlite3.Connection | None = None
        if use_cache:
            self.cache = sqlite3.connect("prompt_cache.db")

    def unwrap_class_answer(
        self,
        answer: str,
        options: Iterable[SCTDescription] = (),
        escape_hatch: SCTDescription | None = EscapeHatch.WORD,
    ) -> SCTDescription | EscapeHatch:
        """\
Check if answer has exactly one valid option.

Assumes that the answer is a valid option if it is wrapped in brackets.
"""
        answer = answer.strip().splitlines()[-1]

        if not options:
            # Return the answer in brackets, if there is one
            if answer.count("[") == answer.count("]") == 1:
                start = answer.index("[") + 1
                end = answer.index("]")
                return SCTDescription(answer[start:end])
            else:
                raise PrompterError(
                    "Could not find a unique option in the answer:", answer
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
            counts[option] = answer.count(option)

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
            option: answer.rfind(wrapped)
            for wrapped, option in wrapped_options.items()
        }
        if all(index == -1 for index in indices.values()):
            raise PrompterError(
                "Could not find a unique option in the answer:", answer
            )

        return max(indices, key=lambda k: indices.get(k, -1))

    def unwrap_bool_answer(
        self,
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
    def ping(self) -> bool:
        """\
Check if the API is available.
"""


class HumanPrompter(Prompter):
    """\
A test prompter that interacts with a meatbag to get answers.

TODO: Interface with a shock collar.
"""

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

        # Get the answer
        while True:
            print(prompt.prompt_message)
            brain_answer = input("Answer: ").strip()
            try:
                return self.unwrap_class_answer(
                    brain_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
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

        while True:
            print(prompt.prompt_message)
            brain_answer = input("Answer: ").strip()
            try:
                return self.unwrap_bool_answer(brain_answer)
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

        while True:
            print(prompt.prompt_message)
            brain_answer = input("Answer: ").strip()
            try:
                return self.unwrap_class_answer(
                    brain_answer,
                    options,
                    EscapeHatch.WORD if allow_escape else None,
                )
            except PrompterError as e:
                print("Error:", e)
                print("Please, provide an answer in the requested format.")

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
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version or self.DEFAULT_VERSION,
        )
        self._endpoint = azure_endpoint
        self._model = model or self.DEFAULT_MODEL

    def ping(self) -> bool:
        print("Pinging the Azure API...")
        try:
            response = requests.get(self._endpoint, timeout=5)
        except requests.ConnectTimeout:
            print("Connection timed out")
            return False
        print("Response:", response.status_code)
        return response.ok

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


class SnowstormAPI:
    TARGET_CODESYSTEM = "SNOMEDCT"

    def __init__(self, url: URL):
        self.url: URL = url
        self.branch_path: BranchPath = self.get_main_branch_path()

    def get_version(self) -> str:
        response = requests.get(
            self.url + "version", headers={"Accept": "application/json"}
        )
        if not response.ok:
            raise SnowstormRequestError.from_response(response)
        return response.json()["version"]

    def get_main_branch_path(self) -> BranchPath:
        # Get codesystems and look for a target
        response = requests.get(
            self.url + "codesystems", headers={"Accept": "application/json"}
        )
        if not response.ok:
            raise SnowstormRequestError.from_response(response)

        for codesystem in response.json()["items"]:
            if codesystem["shortName"] == self.TARGET_CODESYSTEM:
                # TODO: double-check by module contents
                return BranchPath(codesystem["branchPath"])

        raise SnowstormAPIError(
            f"Target codesystem {self.TARGET_CODESYSTEM} is not present"
        )

    def get_branch_info(self) -> dict:
        response = requests.get(
            self.url + f"branches/{self.branch_path}",
            headers={"Accept": "application/json"},
        )
        if not response.ok:
            raise SnowstormRequestError.from_response(response)

        return response.json()

    def get_attribute_suggestions(
        self, parent_ids: Iterable[SCTID]
    ) -> list[AttributeConstraints]:
        params = {
            "parentIds": [*parent_ids],
            "proximalPrimitiveModeling": True,  # Maybe?
            # Filter post-coordination for now
            "contentType": "NEW_PRECOORDINATED",
        }

        response = requests.get(
            url=self.url + "mrcm/" + self.branch_path + "/domain-attributes",
            params=params,
            headers={"Accept": "application/json"},
        )
        if not response.ok:
            raise SnowstormRequestError.from_response(response)

        return [
            AttributeConstraints.from_json(attr)
            for attr in response.json()["items"]
        ]

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
        total = 0
        offset = 0
        step = 100
        collected_items = []

        while offset < total or total == 0:
            response = requests.get(
                url=self.url + f"/{self.branch_path}/members",
                params={
                    "referenceSet": MRCM_DOMAIN_REFERENCE_SET_ECL,
                    "active": True,
                    "offset": offset,
                    "limit": step,
                },
                headers={"Accept": "application/json"},
            )
            if not response.ok:
                raise SnowstormRequestError.from_response(response)

            collected_items.extend(response.json()["items"])
            total = response.json()["total"]
            offset += step

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

        return parents

    def get_attribute_values(
        self, ancestors: Iterable[SCTID], attribute: SCTID
    ) -> dict[SCTID, SCTDescription]:
        # First, obtain the range constraints
        params = {
            "parentIds": [*ancestors],
            "proximalPrimitiveModeling": True,  # Maybe?
            "contentType": "NEW_PRECOORDINATED",
        }

        response = requests.get(
            url=self.url + "mrcm/" + self.branch_path + "/domain-attributes",
            params=params,
            headers={"Accept": "application/json"},
        )

        if not response.ok:
            raise SnowstormRequestError.from_response(response)

        # Find the attribute in the response
        for attr in response.json()["items"]:
            if SCTID(attr["conceptId"]) == attribute:
                attribute_data = attr
                break
        else:
            # Attribute has no valid range for selected ancestors
            return {}

        print("Attribute data:")
        print(json.dumps(attribute_data, indent=2))

        constraint = ECLExpression(
            attribute_data["attributeRange"][0]["rangeConstraint"]
        )

        values: dict[SCTID, SCTDescription] = self._range_constraint_to_parents(
            constraint
        )
        return values

    def get_pt(self, id: SCTID) -> SCTDescription:
        response = requests.get(
            url=self.url + f"concepts/{id}",
            headers={"Accept": "application/json"},
        )
        if not response.ok:
            raise SnowstormRequestError.from_response(response)

        return SCTDescription(response.json()["pt"]["term"])


# Main logic host
class Bouzyges:
    """\
    Main logic host for the Bouzyges system.
    """

    def __init__(
        self,
        snowstorm: SnowstormAPI,
        prompter: Prompter,
        mrcm_entry_points: Iterable[MRCMDomainRefsetEntry],
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

        # Load MRCM entries
        self.mrcm_entries = mrcm_entry_points

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
            print("Possible attributes for:", term)
            attributes = self.snowstorm.get_attribute_suggestions(
                portrait.ancestor_anchors
            )

            # Remove previously rejected attributes
            attributes = [
                attribute
                for attribute in attributes
                if attribute.sctid not in portrait.rejected_attributes
            ]

            # Confirm the attributes
            for attribute in attributes:
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

    def populate_unchecked_attributes(self) -> None:
        for portrait in self.portraits.values():
            rejected = set()
            for attribute in portrait.unchecked_attributes:
                # Get possible attribute values
                values_options = self.snowstorm.get_attribute_values(
                    ancestors=portrait.ancestor_anchors,
                    attribute=attribute,
                )
                print("Attribute:", attribute)
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
                    attribute=snowstorm.get_pt(attribute),
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


if __name__ == "__main__":
    # Load environment variables for API access
    dotenv.load_dotenv()

    snowstorm_endpoint = os.getenv("SNOWSTORM_ENDPOINT")
    if not snowstorm_endpoint:
        raise ValueError("SNOWSTORM_ENDPOINT environment variable is not set")
    snowstorm = SnowstormAPI(URL(snowstorm_endpoint))

    print("Snowstorm Verson:" + snowstorm.get_version())
    print("Using branch path:", snowstorm.branch_path)

    print("MRCM Domain Reference Set entries:")
    domain_entries = snowstorm.get_mrcm_domain_reference_set_entries()
    print("Total entries:", len(domain_entries))
    domain_entry_points = [
        MRCMDomainRefsetEntry.from_json(entry)
        for entry in domain_entries
        if SCTID(entry["referencedComponent"]["conceptId"])
        in WHITELISTED_SUPERTYPES
    ]

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
            print("Azure API is not available, using human prompter.")
            prompter = HumanPrompter(prompt_format=DefaultPromptFormat())
    else:
        print("No Azure API key found, using human prompter.")
        prompter = HumanPrompter(prompt_format=DefaultPromptFormat())

    # Test
    term = "Pyogenic liver abscess"

    bouzyges = Bouzyges(
        snowstorm=snowstorm,
        prompter=prompter,
        mrcm_entry_points=domain_entry_points,
        terms=[term],
    )

    bouzyges.populate_attribute_candidates()
    bouzyges.populate_unchecked_attributes()
