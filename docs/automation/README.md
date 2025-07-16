# Automation Scripts Guide

## Contents
- [Introduction](#introduction)
- [clean.sh Script](#cleansh-script)
- [test.sh Script](#testsh-script)
- [Best Practices](#best-practices)

---

## Introduction
This project provides automation scripts to simplify common development tasks and improve efficiency.

---

## clean.sh Script
- Purpose: Clean build artifacts, caches, and temp files.
- Usage:
  ```bash
  ./clean.sh
  ```
- Recommended before major changes or releases.

---

## test.sh Script
- Purpose: Run all unit tests and code checks with one command.
- Usage:
  ```bash
  ./test.sh
  ```
- Can be used to simulate CI locally.

---

## Best Practices
- Make scripts executable: `chmod +x clean.sh test.sh`
- Integrate with CI/CD or pre-commit hooks.
- If customizing, keep original functionality as fallback.
