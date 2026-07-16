"""The runnable documentation stays runnable.

The README's Quick start blocks and the shipped example execute as written, so
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


def _quick_start_blocks() -> list[str]:
    text = README.read_text()
    section = re.search(r"## Quick start\n(.*?)\n## ", text, re.S)
    assert section, "README Quick start section not found"
    blocks = re.findall(r"```python\n(.*?)```", section.group(1), re.S)
    assert blocks, "no python blocks in Quick start"
    return blocks


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
    ns: dict = {}  # blocks run cumulatively, as a reader typing along would
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(_quick_start_blocks()):
            exec(compile(block, f"<README quick start {i}>", "exec"), ns)


@pytest.mark.slow
def test_example_script_runs(in_tmp_cwd):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runpy.run_path(
            str(ROOT / "examples" / "basic_matching.py"), run_name="__main__"
        )
