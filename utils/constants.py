from .types import PrompterOption, SCTID, ECLExpression, EscapeHatch
import csv

DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_PROMPTERS: dict[PrompterOption, str] = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "human": "Human",
}
CSV_SEPARATORS: dict[str, str] = {
    ",": ",",
    ";": ";",
    "Tab": "\t",
}
QUOTECHARS: list[str] = ['"', "'"]
QUOTING_POLICY: dict[int, str] = {
    csv.QUOTE_MINIMAL: "Quote minimal",
    csv.QUOTE_ALL: "Quote all",
    csv.QUOTE_NONNUMERIC: "Quote string",
    csv.QUOTE_NONE: "No quoting",
}


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

### Default prompt repetition count
DEFAULT_REPEAT_PROMPTS: int | None = 3
