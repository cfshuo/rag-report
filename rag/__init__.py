# -*- coding: utf-8 -*-
"""
rag — 海洋工程 RAG 知识库核心业务包

子模块:
    cleaner   — 文档清洗 (Docling + VLM)
    extractor — 元数据提取 (LLM)
    vectordb  — 向量数据库构建 (Chroma + Embedding)
    utils     — 公共工具 (LM Studio 控制、日志、哈希)
"""

from rag import cleaner, extractor, vectordb

__all__ = ["cleaner", "extractor", "vectordb"]
