"""The runnable documentation stays runnable.

The README's Quick start block and the shipped example execute as written, so
a reader who copies them gets working code (this catches drift like a renamed
method). The README's other snippets are illustrative: they reference
covariates and data the reader supplies, so they are not executed here.
"""

import os
import re
import runpy
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"


def _quick_start_block() -> str:
    text = README.read_text()
    m = re.search(r"## Quick start.*?```python\n(.*?)```", text, re.S)
    assert m, "README Quick start python block not found"
    return m.group(1)


@pytest.fixture
def in_tmp_cwd(tmp_path):
    """Run in a scratch directory; the docs write files (supplement, plots)."""
    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(prev)


def test_readme_quick_start_runs(in_tmp_cwd):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(_quick_start_block(), "<README quick start>", "exec"), {})


@pytest.mark.slow
def test_example_script_runs(in_tmp_cwd):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runpy.run_path(
            str(ROOT / "examples" / "basic_matching.py"), run_name="__main__"
        )
