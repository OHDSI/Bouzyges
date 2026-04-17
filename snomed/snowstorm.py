import asyncio
import datetime
import itertools
import json
import re
from collections.abc import Iterable
from typing import Mapping

import httpx

from models import (
    AttributeConstraints,
    AttributeRelationship,
    Concept,
    MRCMDomainRefsetEntry,
    SemanticPortrait,
)
from utils.constants import (
    IS_A,
    MRCM_DOMAIN_REFERENCE_SET_ECL,
    ROOT_CONCEPT,
    WHITELISTED_SUPERTYPES,
)
from utils.decorators import retry_fixed
from utils.exceptions import (
    SnowstormAPIError,
    SnowstormRequestError,
)
from utils.logger import LOGGER
from utils.types import (
    SCTID,
    BranchPath,
    ECLExpression,
    JsonPrimitive,
    SCTDescription,
    Url,
)


class SnowstormAPI:
    TARGET_CODESYSTEM = "SNOMEDCT"
    CONTENT_TYPE_PREFERENCE = "NEW_PRECOORDINATED", "PRECOORDINATED", "ALL"
    PAGINATION_STEP = 100
    MAX_BAD_PARENT_QUERY = 32

    def __init__(self, url: Url, http_client: httpx.AsyncClient | None = None):
        # Debug
        self.__start_time = datetime.datetime.now()
        self.logger = LOGGER.getChild(self.__class__.__name__)
        self.url: Url = url
        self.mrcm_entries: list[MRCMDomainRefsetEntry] = []

        self.logger.debug(
            f"Snowstorm API URL: {self.url}; initializing async client"
        )
        self.async_client = http_client or httpx.AsyncClient()
        self.logger.info("Snowstorm API client initialized")

        # Cache repetitive queries
        self.__concepts_cache: dict[SCTID, Concept] = {}
        self.__subsumptions_cache: dict[tuple[SCTID, SCTID], bool] = {}

    @classmethod
    async def init(
        cls, url: Url, http_client: httpx.AsyncClient
    ) -> SnowstormAPI:
        snowstorm = cls(url, http_client)

        snowstorm.logger.info("Testing connection...")
        await snowstorm.ping()
        snowstorm.logger.info("Connection successful")

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
        self.logger.debug("Getting Snowstorm version and branch path")
        # Get the main branch path (and try connecting)
        try:
            self.logger.info("Snowstorm Version: " + await self.get_version())
            self.branch_path: BranchPath = await self.get_main_branch_path()
        except Exception as e:
            self.logger.error(
                f"Could not get branch from Snowstorm API: {e}", exc_info=e
            )
            raise
        # Log the version
        self.logger.info("Using branch path: " + self.branch_path)
        return True

    async def _get(self, *args, **kwargs) -> httpx.Response:
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

        try:
            response = await self._get_with_retries(*args, **kwargs)
        except Exception as e:
            self.logger.error(
                f"Could not connect to Snowstorm API: {e}", exc_info=e
            )
            raise

        if not response.status_code < 400:
            self.logger.error(
                f"Request failed: {response.status_code} for {response.url}"
            )
            self.logger.error(
                f"Response: {json.dumps(response.json(), indent=2)}"
            )
            raise SnowstormRequestError.from_response(response)

        return response

    @retry_fixed
    async def _get_with_retries(self, *args, **kwargs) -> httpx.Response:
        response = await self.async_client.get(*args, **kwargs, timeout=120)
        self.logger.debug("Success for %s", response.url)
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

    async def get_concepts(
        self, sctids: Iterable[SCTID]
    ) -> dict[SCTID, Concept]:
        out: dict[SCTID, Concept] = {}
        sctids = set(sctids)
        in_cache: set[SCTID] = set()
        for sctid in sctids:
            if sctid in self.__concepts_cache:
                in_cache.add(sctid)
                out[sctid] = self.__concepts_cache[sctid]
        sctids -= in_cache

        if sctids:
            response = await self._get_collect(
                url=f"browser/{self.branch_path}/concepts",
                params={"activeFilter": True, "conceptIds": list(sctids)},
            )

            for concept in (Concept.from_json(json_) for json_ in response):
                out[concept.sctid] = concept

        return out

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

        # This function results in modification of the bad_parents set;
        # It also runs asynchronoously and will also read the bad_parents set;
        # So we need to make a copy of the set now.
        # TODO: Reconsider running this function in parallel. It may be faster
        # because the target set will shrink each iteration.
        known_bad_parents = set(bad_parents)

        # Batch request, because Snowstorm hates long urls
        for bad_batch in itertools.batched(
            known_bad_parents, self.MAX_BAD_PARENT_QUERY
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
        # This is actually faster synchronously
        if not await self.is_concept_descendant_of(
            child.attribute, parent.attribute
        ):
            return False

        return await self.is_concept_descendant_of(child.value, parent.value)

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

        await asyncio.gather(*map(check_pair, ancestor_matrix))

        portrait.ancestor_anchors -= redundant_ancestors

    async def get_concept_ppp(self, concept: SCTID) -> set[SCTID]:
        """\
Get a concept's Proximal Primitive Parents
"""
        response = await self._get(
            f"{self.branch_path}/concepts/{concept}/authoring-form",
        )

        out: set[SCTID] = set()

        for parent in response.json()["concepts"]:
            sctid = SCTID(parent["id"])
            if parent["primitive"]:
                out.add(sctid)
            else:
                out |= await self.get_concept_ppp(sctid)

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

        for anchor in portrait.ancestor_anchors:
            if not unmatched_predicate_ppp:
                # All matched, escape early
                continue
            anchor_matched_ppp = set()

            for ppp in unmatched_predicate_ppp:
                if await self.is_concept_descendant_of(anchor, ppp):
                    anchor_matched_ppp.add(ppp)

            unmatched_predicate_ppp -= anchor_matched_ppp

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
            msg = [f"Does not satisfy {unmatched} attribute constraints:"]
            for rel in unmatched_concept_relationships:
                msg.append(f" - {rel.attribute} = {rel.value}")
            self.logger.debug("\n".join(msg))
            return False

        self.logger.debug("All constraints are satisfied")
        if not parent_predicate.defined:
            self.logger.debug(
                "Concept is primitive: subsumption must be confirmed manually"
            )
        return True
