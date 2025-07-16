# API 参考文档

本文件概述 Resume Classifier 项目的主要类、方法及用法示例。

## 目录
- [核心类](#核心类)
- [关键方法](#关键方法)
- [用法示例](#用法示例)
- [自动生成 API 文档建议](#自动生成-api-文档建议)

---

## 核心类
- `ResumeJobMatcher`
- `ClassicModel`
- `TransformerModel`
- `DataLoader`
- `DataProcessor`

## 关键方法
- `train()`
- `predict()`
- `predict_proba()`
- `score()`
- `save()` / `load()`

## 用法示例
```python
from src.resume_classifier import ResumeJobMatcher
matcher = ResumeJobMatcher(model_type="classic")
matcher.train(resumes, jobs, labels)
predictions = matcher.predict(resumes, jobs)
```

## 自动生成 API 文档建议
建议使用 [Sphinx](https://www.sphinx-doc.org/) 的 autodoc 功能，从代码自动生成 HTML API 文档。
