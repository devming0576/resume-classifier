# pre-commit 钩子说明

## 目录
- [简介](#简介)
- [安装与配置](#安装与配置)
- [常用钩子](#常用钩子)
- [常见问题](#常见问题)

---

## 简介
pre-commit 用于在提交代码前自动检查和修复常见问题，提升代码质量。

---

## 安装与配置
1. 安装 pre-commit：
   ```bash
   pip install pre-commit
   ```
2. 安装钩子：
   ```bash
   pre-commit install
   ```
3. 配置文件：`.pre-commit-config.yaml`

---

## 常用钩子
- black：自动格式化 Python 代码
- flake8：静态代码检查
- isort：导入排序
- end-of-file-fixer：修复文件结尾
- trailing-whitespace：去除行尾空格
- detect-aws-credentials：检测 AWS 密钥泄漏

---

## 常见问题
- Q: 提交时钩子失败怎么办？
  A: 按提示修复问题后重新提交，或用 `git commit --no-verify` 跳过（不推荐）。
- Q: 如何更新钩子？
  A: 运行 `pre-commit autoupdate`。
