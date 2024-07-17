import requests
import json


## Typing
class BranchPath(str):
    pass


class SCTID(int):
    pass


class Term(str):
    pass


class ECLExpression(str):
    pass


## Connection constants
SNOWSTORM_API = "http://localhost:8080/"
TARGET_CODESYSTEM = "SNOMEDCT"

## Logic constants
MRCM_DOMAIN_REFERENCE_SET_ECL = ECLExpression("<<723589008")
WHITELISTED_SUPERTYPES: set[SCTID] = {
    SCTID(404684003),  # Clinical finding
    SCTID(71388002),  # Procedure
}


## Classes
class SemanticPortrait:
    def __init__(self, term: str, context: str | None = None) -> None:
        self.term: str = term
        self.context: str | None = context
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
        return cls(
            f"Snowstorm API returned {response.status_code} status code",
            response,
        )


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


def get_mrcm_domain_reference_set_entries(branch_path: BranchPath) -> list[dict]:
    total = 0
    offset = 0
    step = 100
    collected_items = []

    while True:
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
        if offset >= total:
            break

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

    for entry in domain_refset_entries:
        component = entry["referencedComponent"]
        if SCTID(component["conceptId"]) not in WHITELISTED_SUPERTYPES:
            continue
        print(
            f"For concept {component['conceptId']}" + f"| {component['pt']['term']} |:"
        )
        print(json.dumps(entry["additionalFields"], indent=2))
        print()

    # Now print all supertypes
    for entry in domain_refset_entries:
        if SCTID(component["conceptId"]) not in WHITELISTED_SUPERTYPES:
            continue
        component = entry["referencedComponent"]
        print(
            f"For concept {component['conceptId']}" + f"| {component['pt']['term']} |;"
        )
