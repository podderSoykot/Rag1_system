from typing import List, NotRequired, TypedDict


class RAGState(TypedDict):
    query: str
    top_k: int
    use_cache: bool
    show_timing: bool
    docs: NotRequired[List[str]]
    prompt: NotRequired[str]
    answer: NotRequired[str]
    cache_hit: NotRequired[bool]
    timing: NotRequired[dict]
    steps: NotRequired[List[str]]
