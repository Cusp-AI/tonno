from __future__ import annotations


class Config:
    """A set of keyword arguments to pass to a tunable function."""

    def __init__(self, **kwargs: int | float | str | bool) -> None:
        self._values = kwargs

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return dict(self._values)

    @classmethod
    def from_dict(cls, d: dict[str, int | float | str | bool]) -> Config:
        return cls(**d)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"Config({inner})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self._values == other._values

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._values.items())))
