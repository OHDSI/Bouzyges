import requests
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable


## Typing
class BranchPath(str):
    pass


class SCTID(int):
    pass


class Term(str):
    pass


class ECLExpression(str):
    pass


class URL(str):
    pass


## Connection constants
SNOWSTORM_API = URL("http://localhost:8080/")
TARGET_CODESYSTEM = "SNOMEDCT"

## Logic constants
MRCM_DOMAIN_REFERENCE_SET_ECL = ECLExpression("<<723589008")
WHITELISTED_SUPERTYPES: set[SCTID] = {
    SCTID(404684003),  # Clinical finding
    SCTID(71388002),  # Procedure
}


## Dataclasses
@dataclass(frozen=True, slots=True)
class AttributeRelationship:
    """Represents a SNOMED CT attribute relationship."""

    attribute: SCTID
    value: SCTID


@dataclass(frozen=True, slots=True)
class AttributeGroup:
    """Represents a SNOMED CT attribute group."""

    group_id: SCTID
    relationships: frozenset[AttributeRelationship]


@dataclass
class SemanticPortrait:
    """\
Represents an interactively built semantic portrait of a source concept."""

    def __init__(self, term: str, context: str | None = None) -> None:
        self.source_term: str = term
        self.context: Iterable[str] | None = context
        self.ancestor_anchors: set[SCTID] = set()
        self.attributes: dict[SCTID, SCTID] = {}


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
    parent_domain: ECLExpression | None = None
    proximal_primitive_refinement: ECLExpression | None = None
    # Currently uses unparseable extension of ECL grammar,
    # but one day we will use it
    domain_template: ECLExpression

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
        return cls(
            f"Snowstorm API returned {response.status_code} status code",
            response,
        )


class PrompterError(Exception):
    """Raised when the prompter encounters an error."""


## Logic classes
class Prompter(ABC):
    """\
Abstract class for forming and sending prompt requests to the LLM agent.
"""

    @staticmethod
    def wrap_term(term: Term) -> str:
        """Wrap a term in brackets."""
        return f"[{term}]"

    def unwrap_class_answer(
        self, answer: str, options: Iterable[Term] = ()
    ) -> Term:
        """\
Check if answer has exactly one valid option.

Assumes that the answer is a valid option if it is wrapped in brackets.
"""
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

        # Check if there is exactly one option present
        wrapped_options = [self.wrap_term(option) for option in options]
        counts = {}
        for option in wrapped_options:
            counts[option] = answer.count(option)

        if sum(map(bool, counts.values())) != 1:
            raise PrompterError(
                "Could not find a unique option in the answer:", answer
            )

        for option, count in counts.items():
            if count:
                return Term(option[1:-1])

    def unwrap_bool_answer(self, answer: str, yes: str, no: str) -> bool:
        """\
Check if the answer contains a yes or no option.
"""
        words = answer.split()
        if yes in words and no not in words:
            return True
        elif no in words and yes not in words:
            return False
        else:
            raise PrompterError(
                "Could not find an unambiguous boolean answer in the response"
            )

    @abstractmethod
    def prompt_for_ancestor(self, term: Term, options: Iterable[Term]) -> Term:
        """\
Prompt the model to choose the best matching ancestor for a term.
"""

    @abstractmethod
    def prompt_for_attribute_presence(
        self, term: Term, attribute: Term
    ) -> bool:
        """\
Prompt the model to choose if an attribute is present in a term.
"""


def get_version() -> str:
    response = requests.get(
        SNOWSTORM_API + "version", headers={"Accept": "application/json"}
    )
    if response.status_code != 200:
        raise SnowstormRequestError.from_response(response)
    return response.json()["version"]


def get_main_branch_path() -> BranchPath:
    # Get codesystems and look for a target
    response = requests.get(
        SNOWSTORM_API + "codesystems", headers={"Accept": "application/json"}
    )
    if response.status_code != 200:
        raise SnowstormRequestError.from_response(response)

    for codesystem in response.json()["items"]:
        if codesystem["shortName"] == TARGET_CODESYSTEM:
            # TODO: double-check by module contents
            return BranchPath(codesystem["branchPath"])

    raise SnowstormAPIError(
        f"Target codesystem {TARGET_CODESYSTEM} is not present on Snowstorm"
    )


def get_branch_info(branch_path: BranchPath) -> dict:
    response = requests.get(
        SNOWSTORM_API + f"branches/{branch_path}",
        headers={"Accept": "application/json"},
    )
    if response.status_code != 200:
        raise SnowstormRequestError.from_response(response)

    return response.json()


def get_attribute_model_hierarchy(branch_path: BranchPath) -> dict:
    response = requests.get(
        SNOWSTORM_API + f"mrcm/{branch_path}/concept-model-attribute-hierarchy",
        headers={"Accept": "application/json"},
    )
    if response.status_code != 200:
        raise SnowstormRequestError.from_response(response)

    return response.json()


def print_attribute_model_hierarchy(amh: dict) -> None:
    def print_node(node, indent):
        print("  " * indent, f"{node["conceptId"]}:", node["pt"]["term"])
        for child in node.get("children", ()):
            print_node(child, indent + 1)

    print_node(amh, 0)


def get_mrcm_domain_reference_set_entries(
    branch_path: BranchPath,
) -> list[dict]:
    total = 0
    offset = 0
    step = 100
    collected_items = []

    while offset < total:
        response = requests.get(
            url=SNOWSTORM_API + f"/{branch_path}/members",
            params={
                "referenceSet": MRCM_DOMAIN_REFERENCE_SET_ECL,
                "active": True,
                "offset": offset,
                "limit": step,
            },
            headers={"Accept": "application/json"},
        )
        if response.status_code != 200:
            raise SnowstormRequestError.from_response(response)

        collected_items.extend(response.json()["items"])
        total = response.json()["total"]
        offset += step

    return collected_items


if __name__ == "__main__":
    print("Snowstorm Verson:" + get_version())
    branch = get_main_branch_path()
    print("Using branch path:", branch)

    branch_info = get_branch_info(branch)
    # print("Branch Info:")
    # print(json.dumps(branch_info, indent=2))

    attribute_model_hierarchy = get_attribute_model_hierarchy(branch)
    # print("Attribute Model Hierarchy:")
    # print_attribute_model_hierarchy(attribute_model_hierarchy)

    print("MRCM Domain Reference Set entries:")
    domain_refset_entries = get_mrcm_domain_reference_set_entries(branch)
    mrcm_entries = [
        MRCMDomainRefsetEntry.from_json(entry)
        for entry in domain_refset_entries
        if SCTID(entry["referencedComponent"]["conceptId"])
        in WHITELISTED_SUPERTYPES
    ]

    for entry in mrcm_entries:
        print(" -", entry.term + ":")
        print("    -", entry.domain_constraint)
        print("    -", entry.guide_link)
        print()
