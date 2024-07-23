import dotenv
import json
import openai
import os
import requests
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable


# Boilerplate
## Typing
class BranchPath(str):
    """Snowstorm working branch"""


class SCTID(int):
    """SNOMED CT identifier"""


class Term(str):
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

    WORD: Term = Term("[NONE]")


class BooleanAnswer(str):
    """\
Boolean answer constants for prompters for yes/no questions
"""

    YES = Term("[AYE]")
    NO = Term("[NAY]")

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
    term: Term

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
            term=Term(json_data["referencedComponent"]["pt"]["term"]),
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
    pt: Term
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
    pt: Term
    range_constraint: ECLExpression
    attribute_rule: ECLExpression


@dataclass(frozen=True, slots=True)
class AttributeConstraints:
    """\
Represents information about an attribute obtained from a known set of parent 
concepts.
"""

    sctid: SCTID
    pt: Term
    attributeDomain: Iterable[AttributeDomain]
    # May actually not be required
    # because /mrcm/{branch}/attribute-values/ exists
    attributeRange: Iterable[AttributeRange]

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            sctid=SCTID(json_data["conceptId"]),
            pt=Term(json_data["pt"]["term"]),
            attributeDomain=[
                AttributeDomain(
                    sctid=SCTID(json_data["conceptId"]),
                    pt=Term(json_data["pt"]["term"]),
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
                    pt=Term(json_data["pt"]["term"]),
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

    def __init__(self, term: str, context: str | None = None) -> None:
        self.source_term: str = term
        self.context: Iterable[str] | None = context
        self.ancestor_anchors: set[SCTID] = set()
        self.attributes: dict[SCTID, SCTID] = {}


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


# Logic classes
## Logic prompt format classes
class PromptFormat(ABC):
    """\
Abstract class for formatting prompts for the LLM agent.
"""

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
        options: Iterable[Term],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
    ) -> str:
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
        attribute: Term,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> str:
        """\
Format a prompt for the LLM agent to decide if an attribute is present in a term.
"""
        _ = term, attribute, term_context, attribute_context
        raise NotImplementedError

    @classmethod
    def form_attr_value(
        cls,
        term: str,
        attribute: Term,
        options: Iterable[Term],
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
        allow_escape: bool = True,
    ) -> str:
        """\
Format a prompt for the LLM agent to choose the value of an attribute in a term.
"""
        _ = (
            term,
            attribute,
            options,
            term_context,
            options_context,
            allow_escape,
        )
        raise NotImplementedError


class DefaultPromptFormat(PromptFormat):
    """\
Default verbose prompt format for the LLM agent.
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
        options: Iterable[Term],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
    ) -> str:
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
        return prompt

    @classmethod
    def form_attr_presence(
        cls,
        term: str,
        attribute: Term,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> str:
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
        return prompt


class OpenAIPromptFormat(PromptFormat):
    """\
Default prompt format for the OpenAI API.
"""

    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        raise NotImplementedError


## Logic prompter classes
class Prompter(ABC):
    """\
Interfaces prompts to the LLM agent and parses answers.
"""

    def __init__(self, *args, prompt_format: PromptFormat, **kwargs):
        _ = args, kwargs
        self.prompt_format = prompt_format

    def unwrap_class_answer(
        self,
        answer: str,
        options: Iterable[Term] = (),
        escape_hatch: Term | None = EscapeHatch.WORD,
    ) -> Term | EscapeHatch:
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
                return Term(answer[start:end])
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
                        Term(option[1:-1])
                        if option != escape_hatch
                        else NULL_ANSWER
                    )

        # Return the last encountered option in brackets
        indices: dict[Term | EscapeHatch, int] = {
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
        options: Iterable[Term],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
    ) -> Term | EscapeHatch:
        """\
Prompt the model to choose the best matching proximal ancestor for a term.
"""

    @abstractmethod
    def prompt_attr_presence(
        self,
        term: str,
        attribute: Term,
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
        attribute: Term,
        options: Iterable[Term],
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
        allow_escape: bool = True,
    ) -> Term | EscapeHatch:
        """\
Prompt the model to choose the value of an attribute in a term.
"""


class HumanPrompter(Prompter):
    """\
A test prompter that interacts with a meatbag to get answers.

TODO: Interface with a shock collar.
"""

    def prompt_supertype(
        self,
        term: str,
        options: Iterable[Term],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
    ) -> Term | EscapeHatch:
        # Construct the prompt
        prompt = self.prompt_format.form_supertype(
            term, options, allow_escape, term_context, options_context
        )

        # Get the answer
        while True:
            print(prompt)
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
        attribute: Term,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> bool:
        _ = term, attribute, term_context, attribute_context
        raise NotImplementedError

    def prompt_attr_value(
        self,
        term: str,
        attribute: Term,
        options: Iterable[Term],
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
        allow_escape: bool = True,
    ) -> Term | EscapeHatch:
        _ = (
            term,
            attribute,
            options,
            term_context,
            options_context,
            allow_escape,
        )
        raise NotImplementedError


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
        self._model = model or self.DEFAULT_MODEL

    def prompt_supertype(
        self,
        term: str,
        options: Iterable[Term],
        allow_escape: bool = True,
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
    ) -> Term | EscapeHatch:
        _ = term, options, allow_escape, term_context, options_context
        raise NotImplementedError

    def prompt_attr_presence(
        self,
        term: str,
        attribute: Term,
        term_context: str | None = None,
        attribute_context: str | None = None,
    ) -> bool:
        _ = term, attribute, term_context, attribute_context
        raise NotImplementedError

    def prompt_attr_value(
        self,
        term: str,
        attribute: Term,
        options: Iterable[Term],
        term_context: str | None = None,
        options_context: dict[Term, str] | None = None,
        allow_escape: bool = True,
    ) -> Term | EscapeHatch:
        _ = (
            term,
            attribute,
            options,
            term_context,
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
            print("  " * indent, f"{node["conceptId"]}:", node["pt"]["term"])
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

    mrcm_entries = [
        MRCMDomainRefsetEntry.from_json(entry)
        for entry in domain_entries
        if SCTID(entry["referencedComponent"]["conceptId"])
        in WHITELISTED_SUPERTYPES
    ]

    for entry in mrcm_entries:
        print(" -", entry.term + ":")
        print("    -", entry.domain_constraint)
        print("    -", entry.guide_link)
        print()

    # Test
    portrait = SemanticPortrait("Pyogenic liver abscess")

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
    else:
        print("No Azure API key found, using human prompter.")
        prompter = HumanPrompter(prompt_format=DefaultPromptFormat())

    supertypes = {entry.term: entry.sctid for entry in mrcm_entries}
    supertype = prompter.prompt_supertype(
        term=portrait.source_term,
        options=supertypes.keys(),
        allow_escape=False,
        term_context="; ".join(portrait.context) if portrait.context else None,
    )
    match supertype:
        case Term(term):
            supertype = supertypes[term]
        case _:
            raise ValueError(
                "Should not happen: did the prompter inject the escape hatch?"
            )

    print("Selected supertype:", supertype)
    attributes = snowstorm.get_attribute_suggestions([supertype])
    for attribute in attributes:
        print(" -", attribute.sctid, attribute.pt)
        for dom in attribute.attributeDomain:
            print("    - Domain is:", dom.domain_id)
        for rng in attribute.attributeRange:
            print("    - Range is:", rng.attribute_rule)
