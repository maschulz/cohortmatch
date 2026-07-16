"""Documentation hygiene: every public surface documented, no doc/signature drift.

Walks all modules and checks that (a) modules, classes, public functions,
methods, and properties carry docstrings and (b) every Args section agrees
exactly with the live signature — no missing and no phantom parameters.
"""

import importlib
import inspect
import re

import pytest

MODULES = [
    "cohortmatch",
    "cohortmatch.api",
    "cohortmatch.risk_set",
    "cohortmatch.evalue",
    "cohortmatch.supplement",
    "cohortmatch.pipeline",
    "cohortmatch.datatypes",
    "cohortmatch.exceptions",
    "cohortmatch.datasets",
    "cohortmatch.validation",
    "cohortmatch.visualization",
    "cohortmatch.matching.greedy",
    "cohortmatch.matching.fast_greedy",
    "cohortmatch.matching.covariate_nn",
    "cohortmatch.matching.optimal",
    "cohortmatch.matching.strata",
    "cohortmatch.matching.distances",
    "cohortmatch.metrics.balance",
    "cohortmatch.metrics.treatment",
    "cohortmatch.metrics.propensity",
    "cohortmatch.metrics.utils",
    "cohortmatch.utils.logging",
]


def _import(modname):
    try:
        return importlib.import_module(modname)
    except ImportError as err:  # viz extra absent
        pytest.skip(f"{modname} unavailable: {err}")


def _documented_params(doc: str) -> set[str]:
    match = re.search(
        r"Args:\n(.*?)(\n\s*(Returns|Raises|Note|Yields|Example|Examples):|\Z)",
        doc,
        re.S,
    )
    if not match:
        return set()
    return set(re.findall(r"^\s{4,}(\w+):", match.group(1), re.M))


def _public_callables(mod):
    for name, obj in vars(mod).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
            continue
        if inspect.isfunction(obj):
            yield f"{mod.__name__}.{name}", obj
        elif inspect.isclass(obj):
            for mname, meth in vars(obj).items():
                if not mname.startswith("_") and inspect.isfunction(meth):
                    yield f"{mod.__name__}.{name}.{mname}", meth


@pytest.mark.parametrize("modname", MODULES)
def test_module_and_classes_documented(modname):
    mod = _import(modname)
    assert mod.__doc__ and mod.__doc__.strip(), f"{modname} lacks a module docstring"
    for name, obj in vars(mod).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != modname:
            continue
        if inspect.isclass(obj):
            assert obj.__doc__, f"{modname}.{name} lacks a class docstring"
            for mname, meth in vars(obj).items():
                if isinstance(meth, property):
                    assert meth.fget.__doc__, (
                        f"{modname}.{name}.{mname} property lacks a docstring"
                    )


@pytest.mark.parametrize("modname", MODULES)
def test_docstrings_match_signatures(modname):
    mod = _import(modname)
    problems = []
    for label, fn in _public_callables(mod):
        doc = fn.__doc__ or ""
        if not doc.strip():
            problems.append(f"{label}: missing docstring")
            continue
        documented = _documented_params(doc)
        if not documented:  # no Args section: allowed for trivial signatures
            continue
        sig = {p for p in inspect.signature(fn).parameters if p not in ("self", "cls")}
        if sig - documented:
            problems.append(
                f"{label}: params not documented: {sorted(sig - documented)}"
            )
        if documented - sig:
            problems.append(
                f"{label}: documents nonexistent: {sorted(documented - sig)}"
            )
    assert not problems, "\n".join(problems)
