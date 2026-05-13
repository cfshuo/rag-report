# -*- coding: utf-8 -*-
"""
chat_rag.py — 海洋工程 RAG 对话系统

基于检索增强生成 (RAG) 的海洋水文气象专家问答引擎。
从本地 Chroma 向量库检索相关知识片段，结合大语言模型生成专业回答。

Usage:
    python chat_rag.py
"""

import logging
import readline  # noqa: F401 — 强制激活终端行编辑（退格/方向键/历史）
import re
import sys
import warnings
from pathlib import Path
from typing import Set

# 抑制第三方库的 deprecation 噪音
warnings.filterwarnings("ignore", message=".*LangChain.*")

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config
from rag.utils.logging_config import setup_logging
from rag.utils.lms import load_both_models, unload_models

_log = setup_logging("chat_rag", level=logging.WARNING)

# 初始化 OpenAI 兼容客户端 (指向本地 LM Studio)
_client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)


def _safe_input(prompt: str = "") -> str:
    """带提示的 input，fallback 处理 UTF-8 边界异常。"""
    try:
        return input(prompt)
    except UnicodeDecodeError:
        return input(prompt)

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
    混合检索：向量语义搜索 + 条款编号文本匹配。

    当查询包含条款编号时，直接从全库搜索包含该编号的 chunk，
    再与语义结果合并，确保条款命中优先。
    """
    clause_nums = _extract_clause_numbers(query)
    if not clause_nums:
        return db.similarity_search(query, k=top_k)

    # 条款编号查询：直接从全库文本搜索
    all_docs = db.get()
    clause_hits = []
    for doc_content, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        for num in clause_nums:
            if num in doc_content:
                from langchain_core.documents import Document
                clause_hits.append(Document(page_content=doc_content, metadata=meta))
                break

    if clause_hits:
        _log.info("条款编号 %s 文本命中 %d 个 chunk", clause_nums, len(clause_hits))
        # 条款命中优先，语义结果补充
        semantic_docs = db.similarity_search(query, k=top_k)
        seen = {doc.page_content for doc in clause_hits}
        for doc in semantic_docs:
            if doc.page_content not in seen:
                clause_hits.append(doc)
                seen.add(doc.page_content)
        return clause_hits[:max(top_k, len(clause_hits))]

    _log.info("条款编号 %s 全库未命中，回退纯语义结果", clause_nums)
    return db.similarity_search(query, k=top_k)


def _build_context(docs) -> tuple[str, Set[str]]:
    """从检索文档构建结构化的上下文文本和来源集合。"""
    context_text = ""
    source_files: Set[str] = set()

    for i, doc in enumerate(docs):
        meta = doc.metadata
        doc_type = meta.get("文档类型", "")
        fname = meta.get("来源文件", "未知文件")

        context_text += f"\n【参考资料 {i + 1}】"

        if doc_type == "规范":
            std_name = meta.get("标准名称", "") or _extract_std_name_from_filename(fname)
            std_code = meta.get("标准编号", "")
            context_text += f"\n- 文档类型: 规范"
            if std_code:
                context_text += f"\n- 标准编号: {std_code}"
            context_text += f"\n- 标准名称: {std_name}"
            # 参考来源格式: 《规范名》 (编号)
            ref = f"《{std_name}》"
            if std_code:
                ref += f" ({std_code})"
            source_files.add(ref)
        else:
            proj = meta.get("项目名称", "未知项目")
            year = meta.get("编制年份", "未知年份")
            loc = meta.get("海域位置", "未知位置")
            stage = meta.get("设计阶段", "未知阶段")
            context_text += f"\n- 项目名称: {proj}"
            context_text += f"\n- 编制年份: {year}"
            context_text += f"\n- 海域位置: {loc}"
            context_text += f"\n- 设计阶段: {stage}"
            # 参考来源格式: 项目名, 年份, 海域, 阶段
            source_files.add(f"{proj}, {year}, {loc}, {stage}")

        context_text += f"\n- 来源文件: {fname}"
        context_text += f"\n[正文片段内容:\n{doc.page_content}\n"
        context_text += "-" * 30 + "\n"

    return context_text, source_files


def _extract_std_name_from_filename(filename: str) -> str:
    """从文件名中提取规范名称，如 '《XXX》出版稿2019.md' → 'XXX'。"""
    m = re.search(r"《(.+?)》", filename)
    return m.group(1) if m else filename


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
            user_query = _safe_input("\n工程师提问: ")
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

            print("\n[参考来源]:")
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
