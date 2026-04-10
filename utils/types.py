from frozendict import frozendict
from typing import (
    Literal,
    TypeVar,
)


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

    def __str__(self) -> str:
        return self.WORD


class BooleanAnswer(str):
    """\
Boolean answer constants for prompters for yes/no questions
"""

    YES = SCTDescription("[AYE]")
    NO = SCTDescription("[NAY]")

    def __new__(cls, value: bool):
        return cls.YES if value else cls.NO


PrompterOption = Literal["human", "openai", "azure"]
JsonPrimitive = int | float | str | bool | None
Json = dict[str, "Json"] | list["Json"] | JsonPrimitive
JsonDict = dict[str, Json]
OpenAIPromptRole = Literal["user", "system", "assisstant"]
OpenAIMessages = tuple[frozendict[OpenAIPromptRole, str]]
OutFormat = Literal["SCG", "CRS", "JSON"]
T = TypeVar("T")
Url = str
