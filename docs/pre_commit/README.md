# pre-commit Hooks Guide

## Contents
- [Introduction](#introduction)
- [Installation & Setup](#installation--setup)
- [Common Hooks](#common-hooks)
- [FAQ](#faq)

---

## Introduction
pre-commit is used to automatically check and fix common issues before committing code, improving code quality.

---

## Installation & Setup
1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```
2. Install hooks:
   ```bash
   pre-commit install
   ```
3. Config file: `.pre-commit-config.yaml`

---

## Common Hooks
- black: auto-format Python code
- flake8: static code analysis
- isort: import sorting
- end-of-file-fixer: fix file endings
- trailing-whitespace: remove trailing spaces
- detect-aws-credentials: detect AWS key leaks

---

## FAQ
- Q: What if a hook fails on commit?
  A: Fix the issue as prompted and commit again, or use `git commit --no-verify` to skip (not recommended).
- Q: How to update hooks?
  A: Run `pre-commit autoupdate`.
