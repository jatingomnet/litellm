from typing import Dict

from typing_extensions import TypedDict


class FastRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: Dict
