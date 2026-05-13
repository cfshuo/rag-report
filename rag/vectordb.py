# -*- coding: utf-8 -*-
"""
vectordb.py — 向量数据库构建器

将清洗后的 Markdown 文档按标题语义切块，调用本地 Embedding 模型向量化，
存入 Chroma 向量数据库。支持增量更新：仅处理新增或修改的文件。

Usage:
    python vector_database_builder.py              # 智能增量构建
    python vector_database_builder.py --full       # 强制全量重建
"""

import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# LangChain 核心组件
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config
from rag.utils.logging_config import setup_logging
from rag.utils.lms import load_model, unload_all, stop_server
from rag.utils.hashing import compute_file_hash, get_changed_files, save_hash_cache

_log = setup_logging("rag.vectordb")

# 扫描的清洗目录 → 文档类型映射
_INPUT_DIRS_TYPE = {
    config.CLEAN_REPORT_DIR: "报告",
    config.CLEAN_STANDARD_DIR: "规范",
}


def _collect_markdown_files() -> list[tuple[Path, str]]:
    """收集所有待处理的 Markdown 文件及其文档类型。"""
    result: list[tuple[Path, str]] = []
    for folder, doc_type in _INPUT_DIRS_TYPE.items():
        if folder.exists():
            for file_path in folder.rglob("*.md"):
                result.append((file_path, doc_type))
    return result


def _load_metadata(md_file: Path) -> dict:
    """加载 Markdown 文件对应的 JSON 元数据。"""
    json_file = md_file.with_suffix(".json")
    if not json_file.exists():
        return {}
    try:
        return json.loads(json_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _log.warning("无法读取 %s 的元数据: %s", json_file.name, e)
        return {}


def _slice_document(md_file: Path, doc_type: str) -> list[Document]:
    """
    对单个文档执行两级切分:
      1. Markdown 标题语义切分
      2. 超长块递归字符切分

    每个切块附带元数据标签（JSON 提取字段 + 来源文件 + 文档类型）。
    """
    content = md_file.read_text(encoding="utf-8")
    metadata = _load_metadata(md_file)
    metadata["来源文件"] = md_file.name
    metadata["文档类型"] = doc_type

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=config.HEADERS_TO_SPLIT_ON
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    md_header_splits = markdown_splitter.split_text(content)
    final_splits = text_splitter.split_documents(md_header_splits)

    for chunk in final_splits:
        # 将标题路径写回正文，确保条款编号可被文本匹配命中
        header_parts = []
        for level in ["Header 1", "Header 2", "Header 3", "Header 4"]:
            h = chunk.metadata.get(level)
            if h:
                header_parts.append(h)
        if header_parts:
            chunk.page_content = " > ".join(header_parts) + "\n\n" + chunk.page_content
        chunk.metadata.update(metadata)

    return final_splits


def _init_embeddings() -> OpenAIEmbeddings:
    """初始化本地 Embedding 接口。"""
    return OpenAIEmbeddings(
        base_url=config.API_BASE_URL,
        api_key=config.API_KEY,
        model=config.EMBEDDING_MODEL_NAME,
        check_embedding_ctx_length=False,
    )


def build_vector_database_full() -> None:
    """
    全量构建向量数据库。

    扫描所有 Markdown 文件，切块嵌入后写入 Chroma。
    适用于首次构建或 --full-rebuild 场景。
    """
    md_files = _collect_markdown_files()
    if not md_files:
        _log.warning("未找到任何 Markdown 文件")
        print("未找到任何 Markdown 文件，请确认清洗步骤已完成。")
        return

    print(f"\n共发现 {len(md_files)} 份文档，开始【语义切块】与【元数据绑定】...")

    all_chunks: list[Document] = []
    hash_cache: dict[str, str] = {}

    for md_file, doc_type in tqdm(md_files, desc="文档切块中", unit="份"):
        chunks = _slice_document(md_file, doc_type)
        all_chunks.extend(chunks)
        # 记录哈希用于后续增量更新
        file_hash = compute_file_hash(md_file)
        if file_hash:
            hash_cache[str(md_file)] = file_hash

    print(f"切块完成！共切出 {len(all_chunks)} 个知识碎片。")
    print(f"\n开始调用 GPU 进行【向量化 (Embedding)】并存入本地数据库...")

    embeddings = _init_embeddings()
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )

    # 保存哈希缓存供后续增量更新使用
    save_hash_cache(config.EMBEDDING_CACHE_FILE, hash_cache)

    print(f"恭喜！向量数据库构建完成！")
    print(f"您的所有知识都已安全保存在本地目录: {config.CHROMA_DB_DIR}")
    _log.info("全量构建完成，共 %d 个块", len(all_chunks))


def build_vector_database_incremental() -> None:
    """
    增量构建向量数据库。

    仅处理新增和修改的文件，跳过未变更的文件。
    首次运行或缓存缺失时自动退化为全量构建。
    """
    md_files = _collect_markdown_files()
    if not md_files:
        _log.warning("未找到任何 Markdown 文件")
        print("未找到任何 Markdown 文件，请确认清洗步骤已完成。")
        return

    # 检查是否有已有数据库
    db_exists = Path(config.CHROMA_DB_DIR).exists() and any(
        Path(config.CHROMA_DB_DIR).iterdir()
    )
    cache_exists = config.EMBEDDING_CACHE_FILE.exists()

    if not db_exists or not cache_exists:
        _log.info("数据库或缓存不存在，执行全量构建")
        build_vector_database_full()
        return

    new_files, modified_files, unchanged_files = get_changed_files(
        [fp for fp, _ in md_files], config.EMBEDDING_CACHE_FILE
    )

    changed = new_files + modified_files
    if not changed:
        print("所有文档与向量数据库一致，无需更新。")
        _log.info("增量构建: 无变更文件，跳过")
        return

    print(f"\n检测到文件变更: 新增 {len(new_files)} 份，修改 {len(modified_files)} 份")
    print(f"未变更 (跳过): {len(unchanged_files)} 份")

    # 构建 {文件路径: (Path, doc_type)} 映射
    file_map = {str(fp): (fp, dt) for fp, dt in md_files}

    all_chunks: list[Document] = []
    new_hash_cache: dict[str, str] = {}

    for file_path_str in tqdm(changed, desc="处理变更文档", unit="份"):
        fp, doc_type = file_map[file_path_str]
        chunks = _slice_document(fp, doc_type)
        all_chunks.extend(chunks)
        file_hash = compute_file_hash(fp)
        if file_hash:
            new_hash_cache[file_path_str] = file_hash

    # 保留未变更文件的哈希
    for fp in unchanged_files:
        file_hash = compute_file_hash(fp)
        if file_hash:
            new_hash_cache[str(fp)] = file_hash

    if not all_chunks:
        _log.info("变更文件无有效内容")
        return

    print(f"切出 {len(all_chunks)} 个新知识碎片，开始向量化...")

    embeddings = _init_embeddings()
    vectorstore = Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(all_chunks)

    save_hash_cache(config.EMBEDDING_CACHE_FILE, new_hash_cache)

    print(f"增量更新完成！新增 {len(all_chunks)} 个向量到数据库。")
    _log.info("增量构建完成，新增 %d 个块", len(all_chunks))


def build_vector_database() -> None:
    """
    智能构建向量数据库入口。

    自动判断使用增量还是全量构建策略。
    """
    db_path = Path(config.CHROMA_DB_DIR)
    if db_path.exists() and config.EMBEDDING_CACHE_FILE.exists():
        build_vector_database_incremental()
    else:
        build_vector_database_full()


if __name__ == "__main__":
    full_rebuild = "--full" in sys.argv

    try:
        load_model("embedding", gpu_ratio="max")
        if full_rebuild:
            print("强制执行全量重建...")
            build_vector_database_full()
        else:
            build_vector_database()
    except Exception as main_e:
        _log.exception("程序中断: %s", main_e)
        print(f"\n程序中断: {main_e}")
    finally:
        print("\n清理战场...")
        unload_all()
        stop_server()
