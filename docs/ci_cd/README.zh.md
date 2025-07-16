# CI/CD 说明与 Badge

## 目录
- [简介](#简介)
- [CI Badge 说明](#ci-badge-说明)
- [CI/CD 流程简介](#cicd-流程简介)
- [如何查看和配置 CI](#如何查看和配置-ci)
- [常见问题](#常见问题)

---

## 简介
本项目采用持续集成/持续交付（CI/CD）流程，自动化测试、格式检查和部署，提升代码质量和协作效率。

---

## CI Badge 说明
README 顶部的 CI Badge 显示主分支的构建状态：

```
![CI](https://github.com/your-org/resume-classifier/actions/workflows/ci.yml/badge.svg)
```

- 绿色（passing）：所有自动化检查通过。
- 红色（failing）：有检查未通过，请点击 badge 查看详情。

---

## CI/CD 流程简介
- 代码提交或 PR 时自动触发。
- 步骤包括依赖安装、代码格式检查、静态分析、单元测试等。
- 通过后可自动部署或发布。

---

## 如何查看和配置 CI
- 配置文件位于 `.github/workflows/ci.yml`。
- 可在 GitHub Actions 页面查看每次运行详情。
- 如需自定义流程，请编辑 ci.yml 文件。

---

## 常见问题
- Q: CI 失败怎么办？
  A: 点击 badge 或 Actions 查看日志，修复对应问题后重新提交。
- Q: 如何本地模拟 CI？
  A: 运行 `./test.sh` 脚本。
