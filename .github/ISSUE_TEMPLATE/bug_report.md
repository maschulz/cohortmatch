---
name: Bug report
about: A result you believe is wrong, or a crash
labels: bug
---

**What happened**

**What you expected**

**Reproducer** (a small, runnable snippet; synthetic data is fine)

```python
```

**Versions**

```
python -c "import cohortmatch, numpy, pandas, sklearn, statsmodels; print(cohortmatch.__version__, numpy.__version__, pandas.__version__, sklearn.__version__, statsmodels.__version__)"
```

**If it is a numerical disagreement with MatchIt or another tool**, include the
other tool's version and the exact call, so the two can be compared directly.
