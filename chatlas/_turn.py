from __future__ import annotations

from typing import Generic, Literal, Optional, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ._content import Content, ContentText, ContentUnion, create_content

__all__ = ("Turn",)

CompletionT = TypeVar("CompletionT")


class Turn(BaseModel, Generic[CompletionT]):
    """
    A user or assistant turn

    Every conversation with a chatbot consists of pairs of user and assistant
    turns, corresponding to an HTTP request and response. These turns are
    represented by the `Turn` object, which contains a list of
    [](`~chatlas.types.Content`)s representing the individual messages within the
    turn. These might be text, images, tool requests (assistant only), or tool
    responses (user only).

    Note that a call to `.chat()` and related functions may result in multiple
    user-assistant turn cycles. For example, if you have registered tools, chatlas
    will automatically handle the tool calling loop, which may result in any
    number of additional cycles.

    Examples
    --------

    ```python
    from chatlas import Turn, ChatOpenAI, ChatAnthropic

    chat = ChatOpenAI()
    str(chat.chat("What is the capital of France?"))
    turns = chat.get_turns()
    assert len(turns) == 2
    assert isinstance(turns[0], Turn)
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"

    # Load context into a new chat instance
    chat2 = ChatAnthropic()
    chat2.set_turns(turns)
    turns2 = chat2.get_turns()
    assert turns == turns2
    ```

    Parameters
    ----------
    role
        Either "user", "assistant", or "system".
    contents
        A list of [](`~chatlas.types.Content`) objects.
    tokens
        A numeric vector of length 3 representing the number of input, output, and cached
        tokens (respectively) used in this turn. Currently only recorded for
        assistant turns.
    finish_reason
        A string indicating the reason why the conversation ended. This is only
        relevant for assistant turns.
    completion
        The completion object returned by the provider. This is useful if there's
        information returned by the provider that chatlas doesn't otherwise expose.
        This is only relevant for assistant turns.
    """

    role: Literal["user", "assistant", "system"]
    contents: list[ContentUnion] = Field(default_factory=list)
    tokens: Optional[tuple[int, int, int]] = None
    finish_reason: Optional[str] = None
    completion: Optional[CompletionT] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        role: Literal["user", "assistant", "system"],
        contents: str | Sequence[Content | str],
        *,
        tokens: Optional[tuple[int, int, int]] = None,
        finish_reason: Optional[str] = None,
        completion: Optional[CompletionT] = None,
        **kwargs,
    ):
        if isinstance(contents, str):
            contents = [ContentText(text=contents)]

        contents2: list[Content] = []
        for x in contents:
            if isinstance(x, Content):
                contents2.append(x)
            elif isinstance(x, str):
                contents2.append(ContentText(text=x))
            elif isinstance(x, dict):
                contents2.append(create_content(x))
            else:
                raise ValueError("All contents must be Content objects or str.")

        super().__init__(
            role=role,
            contents=contents2,
            tokens=tokens,
            finish_reason=finish_reason,
            completion=completion,
            **kwargs,
        )

    @property
    def text(self) -> str:
        return "".join(x.text for x in self.contents if isinstance(x, ContentText))

    def get_cache_info(self) -> dict[str, int | float | None]:
        """
        Get cache performance information for this turn.

        Returns
        -------
        dict
            A dictionary with cache performance metrics including:
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
            - cached_tokens: Number of tokens read from cache
            - cache_hit_rate: Percentage of input tokens that came from cache (0-100)
            - tokens_saved: Number of tokens saved due to caching
        """
        if not self.tokens:
            return {
                "input_tokens": None,
                "output_tokens": None,
                "cached_tokens": None,
                "cache_hit_rate": None,
                "tokens_saved": None,
            }

        input_tokens, output_tokens, cached_tokens = self.tokens

        # Cache hit rate as percentage
        cache_hit_rate = (
            (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_rate": round(cache_hit_rate, 1),
            "tokens_saved": cached_tokens,
        }

    def __str__(self) -> str:
        return self.text

    def __repr__(self, indent: int = 0) -> str:
        res = " " * indent + f"<Turn role='{self.role}'"
        if self.tokens:
            input_tokens, output_tokens, cached_tokens = self.tokens
            res += f" tokens={self.tokens}"
            # Add cache performance info if there are cached tokens
            if cached_tokens > 0:
                cache_info = self.get_cache_info()
                res += f" cache_hit_rate={cache_info['cache_hit_rate']}%"
        if self.finish_reason:
            res += f" finish_reason='{self.finish_reason}'"
        if self.completion:
            res += f" completion={self.completion}"
        res += ">"
        for content in self.contents:
            res += "\n" + content.__repr__(indent=indent + 2)
        return res + "\n"


def user_turn(*args: Content | str) -> Turn:
    if len(args) == 0:
        raise ValueError("Must supply at least one input.")

    return Turn("user", args)
