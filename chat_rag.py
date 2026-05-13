# -*- coding: utf-8 -*-
"""
chat_rag.py — 海洋工程 RAG 对话系统

基于检索增强生成 (RAG) 的海洋水文气象专家问答引擎。
从本地 Chroma 向量库检索相关知识片段，结合大语言模型生成专业回答。

Usage:
    python chat_rag.py
"""

import re
import sys
from pathlib import Path
from typing import Set

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import config
from rag.utils.logging_config import setup_logging
from rag.utils.lms import load_both_models, unload_models

_log = setup_logging("chat_rag")

# 初始化 OpenAI 兼容客户端 (指向本地 LM Studio)
_client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)

# 条款编号匹配模式
_CLAUSE_PATTERNS = [
    re.compile(r"第\s*(\d+(?:\.\d+)*)\s*[条章节款]"),
    re.compile(r"[条款章节]\s*(\d+(?:\.\d+)*)"),
]


def _extract_clause_numbers(query: str) -> set[str]:
    """从查询中提取条款编号，如 '4.1.5'、'第4.1.5条'。"""
    numbers: set[str] = set()
    for p in _CLAUSE_PATTERNS:
        numbers.update(p.findall(query))
    return numbers


def _hybrid_retrieval(db, query: str, top_k: int):
    """
    混合检索：向量语义搜索 + 条款编号文本匹配重排序。

    当查询包含条款编号时，扩大召回范围并用文本匹配提升精度。
    """
    clause_nums = _extract_clause_numbers(query)
    fetch_k = max(top_k * 4, 12) if clause_nums else top_k
    docs = db.similarity_search(query, k=fetch_k)

    if not clause_nums:
        return docs[:top_k]

    # 文本匹配加分：chunk 内容命中条款编号的排前面
    scored = []
    for doc in docs:
        score = sum(1 for num in clause_nums if num in doc.page_content)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored]

    if not any(s > 0 for s, _ in scored):
        _log.info("条款编号 %s 未在任何 chunk 中命中，回退纯语义结果", clause_nums)
    return reranked[:top_k]


def _build_context(docs) -> tuple[str, Set[str]]:
    """从检索文档构建结构化的上下文文本和来源集合。"""
    context_text = ""
    source_files: Set[str] = set()

    for i, doc in enumerate(docs):
        proj = doc.metadata.get("项目名称", "未知项目")
        fname = doc.metadata.get("来源文件", "未知文件")
        year = doc.metadata.get("编制年份", "未知年份")
        loc = doc.metadata.get("海域位置", "未知位置")
        stage = doc.metadata.get("设计阶段", "未知阶段")

        context_text += f"\n【参考资料 {i + 1}】"
        context_text += f"\n- 项目名称: {proj}"
        context_text += f"\n- 编制年份: {year}"
        context_text += f"\n- 海域位置: {loc}"
        context_text += f"\n- 设计阶段: {stage}"
        context_text += f"\n- 来源文件: {fname}"
        context_text += f"\n[正文片段内容:\n{doc.page_content}\n"
        context_text += "-" * 30 + "\n"

        source_files.add(f"{proj} ({fname})")

    return context_text, source_files


def _build_system_prompt(context_text: str) -> str:
    """构建系统提示词，包含检索到的参考资料。"""
    return (
        "你是一位专业的海洋水文气象高级工程师。\n"
        "请严谨地基于提供的【参考资料】（包含项目属性和正文片段）来回答用户的问题。\n"
        "资料中写了什么就回答什么，绝对不要编造任何数据。\n\n"
        f"以下是相关参考资料：\n{context_text}"
    )


def chat_loop() -> None:
    """
    RAG 对话主循环。

    挂载本地 Chroma 向量数据库，循环接收用户提问，
    检索相关知识片段并调用 LLM 流式生成回答。
    """
    _log.info("正在挂载本地向量数据库...")
    try:
        embeddings = OpenAIEmbeddings(
            base_url=config.API_BASE_URL,
            api_key=config.API_KEY,
            model=config.EMBEDDING_MODEL_NAME,
            check_embedding_ctx_length=False,
        )
        db = Chroma(
            persist_directory=config.CHROMA_DB_DIR,
            embedding_function=embeddings,
        )
        _log.info("数据库挂载成功")
    except Exception as e:
        _log.error("数据库连接失败: %s", e)
        print(f"数据库连接失败: {e}")
        return

    print("\n" + "=" * 60)
    print("海洋工程智能专家库 (RAG 系统) 已准备就绪")
    print("输入 'exit' 或 'quit' 退出对话")
    print("=" * 60)

    chat_history: list[dict] = []

    while True:
        try:
            user_query = input("\n工程师提问: ")
        except (EOFError, KeyboardInterrupt):
            print("\n用户退出。")
            break

        if user_query.lower() in ("exit", "quit", "退出"):
            break
        if not user_query.strip():
            continue

        print("正在检索相关水文报告与规范片段...")
        docs = _hybrid_retrieval(db, user_query, config.RETRIEVAL_K)

        if not docs:
            _log.info("未检索到相关知识: %s", user_query[:80])
            print("系统: 未在库中找到相关知识。")
            continue

        context_text, source_files = _build_context(docs)
        system_prompt = _build_system_prompt(context_text)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history[-4:])  # 保留最近两轮对话记忆
        messages.append({"role": "user", "content": user_query})

        print("专家回答: \n", end="", flush=True)

        try:
            response = _client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=0.2,
                stream=True,
            )

            full_answer = ""
            for chunk in response:
                choices = getattr(chunk, "choices", None)
                if choices and len(choices) > 0:
                    delta = getattr(choices[0], "delta", None)
                    if delta and getattr(delta, "content", None):
                        content = delta.content
                        print(content, end="", flush=True)
                        full_answer += content

            print("\n\n[参考来源]:")
            for s in sorted(source_files):
                print(f"- {s}")

            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": full_answer})

        except Exception as e:
            _log.error("模型生成异常: %s", e)
            print(f"\n模型生成异常: {e}")


if __name__ == "__main__":
    try:
        load_both_models()
        chat_loop()
    except KeyboardInterrupt:
        print("\n用户手动退出。")
    except Exception as e:
        _log.exception("程序异常退出")
        print(f"\n程序异常: {e}")
    finally:
        unload_models()
