# -*- coding: utf-8 -*-
"""
extractor.py — 文档元数据智能提取

使用 LLM 从清洗后的 Markdown 文档中提取结构化元数据（项目名称、海域位置、
编制年份等），输出为同名 JSON 文件，供向量库构建时绑定到每个知识块。

Usage:
    python information_extract.py
"""

import json
import logging
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import config
from rag.utils.hashing import compute_file_hash, load_hash_cache, save_hash_cache
from rag.utils.logging_config import setup_logging
from rag.utils.lms import load_model, unload_all, stop_server

_log = setup_logging("rag.extractor")

_client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)

# 重试配置
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # 秒，指数增长

# 报告类文档的提取提示词
_PROMPT_REPORT = (
    "你是一个严谨的水文数据提取程序。请阅读文本提取信息。\n"
    "【最高指令】你必须且只能输出包含大括号的纯 JSON 代码！"
    "绝不能有任何开头问候或结尾解释！\n"
    "找不到的信息请填 \"未知\"。\n\n"
    "你必须严格按照以下模板输出：\n"
    + json.dumps(config.METADATA_FIELDS_REPORT, ensure_ascii=False, indent=4)
)

# 标准类文档的提取提示词
_PROMPT_STANDARD = (
    "你是一个严谨的标准审查程序。请阅读文本提取信息。\n"
    "【最高指令】你必须且只能输出包含大括号的纯 JSON 代码！"
    "绝不能有任何开头问候或结尾解释！\n"
    "找不到的信息请填 \"未知\"。\n\n"
    "你必须严格按照以下模板输出：\n"
    + json.dumps(config.METADATA_FIELDS_STANDARD, ensure_ascii=False, indent=4)
)


def _read_document_content(md_file: Path) -> str:
    """
    读取文档内容用于元数据提取。

    根据 config.MAX_TEXT_FOR_METADATA 决定读取策略:
      - None: 读取全文
      - 数值: 取头部 + 尾部各一半，确保首尾关键信息不遗漏
    """
    full_text = md_file.read_text(encoding="utf-8")
    limit = config.MAX_TEXT_FOR_METADATA

    if limit is None or len(full_text) <= limit:
        return full_text

    half = limit // 2
    head = full_text[:half]
    tail = full_text[-half:]
    return head + "\n...(中间内容省略)...\n" + tail


def _parse_json_response(raw_text: str) -> dict:
    """从 LLM 响应中暴力提取 JSON 对象。"""
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not json_match:
        _log.warning("LLM 未输出 JSON 结构，原话: %s", raw_text[:200])
        tqdm.write(f"\n格式彻底失控！大模型原话：\n{raw_text}\n" + "-" * 30)
        return {"解析错误": "大模型未输出大括号结构"}

    clean_json_str = json_match.group(0)
    try:
        return json.loads(clean_json_str)
    except json.JSONDecodeError as jde:
        _log.error("JSON 语法错误: %s，文本: %s", jde, clean_json_str)
        tqdm.write(f"\nJSON 语法错误！抠出的文本：\n{clean_json_str}\n" + "-" * 30)
        return {"解析错误": "大括号内存在语法错误 (如引号缺失)"}


def extract_metadata_from_text(text_chunk: str, doc_type: str) -> dict:
    """
    使用 LLM 从文本中提取结构化元数据。

    Args:
        text_chunk: 文档文本内容
        doc_type: 文档类型 — 'report' 或 'standard'

    Returns:
        提取的元数据字典，失败时返回含 "解析错误" 键的字典
    """
    system_prompt = _PROMPT_REPORT if doc_type == "report" else _PROMPT_STANDARD
    user_prompt = f"阅读以下文本，严格输出 JSON：\n\n{text_chunk}"

    last_error = ""
    for attempt in range(1 + MAX_RETRIES):
        try:
            api_kwargs: dict = {
                "model": config.LLM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "extra_body": {"reasoning_effort": config.REASONING_EFFORT},
            }

            response = _client.chat.completions.create(**api_kwargs)
            result_text = response.choices[0].message.content.strip()
            return _parse_json_response(result_text)

        except Exception as e:
            last_error = str(e)
            _log.warning("API 调用失败 (尝试 %d/%d): %s", attempt + 1, 1 + MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)

    _log.error("API 调用全部重试失败: %s", last_error)
    return {"状态": "调用失败", "错误详情": last_error}


def process_markdown_files(full_rebuild: bool = False) -> None:
    """
    主流程: 扫描清洗目录中的所有 Markdown 文件，逐份调用 LLM 提取元数据，
    结果写入同名 JSON 文件。

    Args:
        full_rebuild: 是否强制全量提取（跳过增量检测）
    """
    input_map = {
        config.CLEAN_REPORT_DIR: "report",
        config.CLEAN_STANDARD_DIR: "standard",
    }

    md_files: list[tuple[Path, str]] = []
    for folder, doc_type in input_map.items():
        if folder.exists():
            for md_file in folder.rglob("*.md"):
                md_files.append((md_file, doc_type))

    if not md_files:
        _log.warning("未找到任何 .md 文件，请检查清洗目录")
        print("未找到任何 .md 文件，请检查清洗文件夹。")
        return

    # 增量检测：跳过未变更的 MD
    old_hashes = {} if full_rebuild else load_hash_cache(config.EXTRACTOR_CACHE_FILE)
    to_process: list[tuple[Path, str]] = []
    skipped = 0
    for md_file, doc_type in md_files:
        json_path = md_file.with_suffix(".json")
        current_hash = compute_file_hash(md_file)
        if not full_rebuild and json_path.exists() and current_hash:
            key = str(md_file)
            if key in old_hashes and old_hashes[key] == current_hash:
                skipped += 1
                continue
        to_process.append((md_file, doc_type))

    if skipped:
        print(f"\n增量模式：跳过 {skipped} 份未变更的文档，待提取 {len(to_process)} 份。")
    else:
        print(f"\n共发现 {len(to_process)} 份文档，开始智能双轨提取...")

    if not to_process:
        print("所有文档已是最新，无需提取。")
        return

    new_hashes: dict[str, str] = {}
    success_count = 0
    pbar = tqdm(to_process, desc="提取进度", unit="份")

    for md_file, doc_type in pbar:
        pbar.set_description(f"读取 ({doc_type}): {md_file.name[:15]}...")
        try:
            content = _read_document_content(md_file)
            metadata = extract_metadata_from_text(content, doc_type)

            json_file_path = md_file.with_suffix(".json")
            json_file_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
            _log.info("提取 %s -> %s", md_file.name, metadata)
            file_hash = compute_file_hash(md_file)
            if file_hash:
                new_hashes[str(md_file)] = file_hash
            success_count += 1
        except Exception as e:
            _log.exception("处理文件崩溃: %s", md_file.name)
            continue

    # 合并哈希缓存
    if not full_rebuild:
        for key, val in old_hashes.items():
            if key not in new_hashes:
                new_hashes[key] = val
    save_hash_cache(config.EXTRACTOR_CACHE_FILE, new_hashes)

    print(f"\n提取完成！成功处理了 {success_count} 份文档。")


if __name__ == "__main__":
    try:
        load_model("llm", gpu_ratio="max")
        process_markdown_files()
    except Exception as main_e:
        _log.exception("程序中断: %s", main_e)
        print(f"\n程序中断: {main_e}")
    finally:
        print("\n清理战场...")
        unload_all()
        stop_server()
