from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Generic, TypeVar

import numpy as np
from scipy.optimize import least_squares

T = TypeVar("T")


def _freeze_token(token: Any) -> tuple[Any, ...]:
    """Convert a variable token into a hashable key."""
    if isinstance(token, np.ndarray):
        arr = np.asarray(token)
        return (arr.dtype.str, arr.shape, tuple(arr.reshape(-1).tolist()))
    if isinstance(token, (list, tuple)):
        arr = np.asarray(token)
        return (arr.dtype.str, arr.shape, tuple(arr.reshape(-1).tolist()))
    return (type(token).__name__, token)


class Var(Generic[T]):
    """Minimal variable handle used by the local PyRoki shim."""

    _default_factory: Callable[[], Any] = lambda: None

    def __init_subclass__(cls, default_factory: Callable[[], Any] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if default_factory is not None:
            cls._default_factory = staticmethod(default_factory)  # type: ignore[assignment]

    def __class_getitem__(cls, _item):
        return cls

    def __init__(self, token: Any, value: Any | None = None):
        self.token = token
        self.id = np.asarray(token)
        self._key = _freeze_token(token)
        self.value = value

    def __hash__(self) -> int:
        return hash((self.__class__, self._key))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self._key == other._key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(token={self.token!r}, value={self.value!r})"

    def with_value(self, value: Any) -> Var[T]:
        return self.__class__(self.token, value)

    def default_factory(self) -> Any:
        return self.__class__._default_factory()


class VarValues:
    """Mapping from variables to concrete values."""

    def __init__(self, values: dict[Var[Any], Any] | None = None):
        self._values: dict[Var[Any], Any] = values or {}

    @classmethod
    def make(cls, vars_: tuple[Var[Any], ...] | list[Var[Any]]) -> VarValues:
        values: dict[Var[Any], Any] = {}
        for var in vars_:
            if var.value is not None:
                values[var] = var.value
            else:
                values[var] = var.default_factory()
        return cls(values)

    def __getitem__(self, var: Var[Any]) -> Any:
        return self._values[var]

    def __setitem__(self, var: Var[Any], value: Any) -> None:
        self._values[var] = value

    def items(self):
        return self._values.items()


@dataclass
class TrustRegionConfig:
    lambda_initial: float = 1.0


class Cost:
    """Minimal cost wrapper compatible with the PyRoki sources in this repo."""

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        kind: str | None = None,
        jac_custom_with_cache_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}
        self.name = name or getattr(fn, "__name__", "Cost")
        self.kind = kind
        self.jac_custom_with_cache_fn = jac_custom_with_cache_fn

    def evaluate(self, vals: VarValues) -> np.ndarray:
        out = self.fn(vals, *self.args, **self.kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            out = out[0]
        return np.asarray(out, dtype=np.float64).reshape(-1)

    @classmethod
    def factory(
        cls,
        fn: Callable[..., Any] | None = None,
        *,
        kind: str | None = None,
        jac_custom_with_cache_fn: Callable[..., Any] | None = None,
        name: str | None = None,
    ):
        def decorator(residual_fn: Callable[..., Any]):
            @wraps(residual_fn)
            def wrapper(*args, **kwargs):
                return cls(
                    residual_fn,
                    args=args,
                    kwargs=kwargs,
                    name=name or residual_fn.__name__,
                    kind=kind,
                    jac_custom_with_cache_fn=jac_custom_with_cache_fn,
                )

            return wrapper

        if fn is None:
            return decorator
        return decorator(fn)

    @classmethod
    def create_factory(cls, fn: Callable[..., Any] | None = None, **kwargs):
        return cls.factory(fn, **kwargs)


class LeastSquaresProblem:
    """Very small nonlinear least-squares driver backed by SciPy."""

    def __init__(self, costs: list[Cost], variables: list[Var[Any]]) -> None:
        self.costs = costs
        self.variables = variables

    def analyze(self) -> LeastSquaresProblem:
        return self

    def _pack(self, vals: VarValues) -> tuple[np.ndarray, list[tuple[Var[Any], tuple[int, ...], int]]]:
        flat_parts: list[np.ndarray] = []
        layout: list[tuple[Var[Any], tuple[int, ...], int]] = []
        for var in self.variables:
            arr = np.asarray(vals[var], dtype=np.float64)
            flat_parts.append(arr.reshape(-1))
            layout.append((var, arr.shape, arr.size))
        return np.concatenate(flat_parts) if flat_parts else np.array([], dtype=np.float64), layout

    def _unpack(
        self,
        x: np.ndarray,
        layout: list[tuple[Var[Any], tuple[int, ...], int]],
    ) -> VarValues:
        values: dict[Var[Any], Any] = {}
        offset = 0
        for var, shape, size in layout:
            values[var] = x[offset : offset + size].reshape(shape)
            offset += size
        return VarValues(values)

    def solve(
        self,
        *,
        verbose: bool | None = None,
        linear_solver: str | None = None,
        trust_region: TrustRegionConfig | None = None,
        initial_vals: VarValues | None = None,
    ) -> VarValues:
        del verbose, linear_solver, trust_region
        if initial_vals is None:
            initial_vals = VarValues.make(tuple(self.variables))
        x0, layout = self._pack(initial_vals)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            vals = self._unpack(x, layout)
            parts = [cost.evaluate(vals) for cost in self.costs]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

        result = least_squares(residual_fn, x0, method="trf")
        return self._unpack(result.x, layout)


__all__ = [
    "Cost",
    "LeastSquaresProblem",
    "TrustRegionConfig",
    "Var",
    "VarValues",
]
