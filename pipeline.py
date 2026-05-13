#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — 海洋工程 RAG 知识库全流程编排脚本

按序自动执行: 文档清洗 → 信息提取 → 向量库构建
支持断点续跑、单步执行、强制全量重建。

Usage:
    python pipeline.py                  # 运行完整流水线
    python pipeline.py --step clean     # 仅运行文档清洗
    python pipeline.py --step extract   # 仅运行信息提取
    python pipeline.py --step build     # 仅运行向量库构建
    python pipeline.py --full-rebuild   # 强制全量重建向量库
    python pipeline.py --dry-run        # 仅打印将要执行的操作
"""

import argparse
import os
import sys
import time

import config
from rag.utils.logging_config import setup_logging
from rag.utils.lms import load_model, unload_all, stop_server

# 设置编码环境变量，确保子进程及 Java 组件采用 UTF-8
os.environ["_JAVA_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dconsole.encoding=UTF-8"
os.environ["PYTHONIOENCODING"] = "utf-8"

_log = setup_logging("pipeline")


def step_clean() -> bool:
    """
    执行文档清洗步骤。

    Returns:
        True 表示成功
    """
    print("\n" + "=" * 60)
    print("  步骤 1/3: 文档清洗 (Docling + VLM)")
    print("=" * 60)
    try:
        from rag import cleaner
        load_model("vlm", gpu_ratio="max")
        cleaner.main(use_vlm=config.USE_VLM)
        unload_all()
        return True
    except Exception as e:
        _log.exception("文档清洗失败: %s", e)
        print(f"\n文档清洗失败: {e}")
        return False


def step_extract() -> bool:
    """
    执行信息提取步骤。

    Returns:
        True 表示成功
    """
    print("\n" + "=" * 60)
    print("  步骤 2/3: 元数据提取 (LLM)")
    print("=" * 60)
    try:
        from rag import extractor
        load_model("llm", gpu_ratio="max")
        extractor.process_markdown_files()
        unload_all()
        return True
    except Exception as e:
        _log.exception("信息提取失败: %s", e)
        print(f"\n信息提取失败: {e}")
        return False


def step_build(full_rebuild: bool = False) -> bool:
    """
    执行向量库构建步骤。

    Args:
        full_rebuild: 是否强制全量重建

    Returns:
        True 表示成功
    """
    print("\n" + "=" * 60)
    print(f"  步骤 3/3: 向量库构建 {'(全量重建)' if full_rebuild else '(智能增量)'}")
    print("=" * 60)
    try:
        from rag import vectordb
        load_model("embedding", gpu_ratio="max")
        if full_rebuild:
            vectordb.build_vector_database_full()
        else:
            vectordb.build_vector_database()
        unload_all()
        return True
    except Exception as e:
        _log.exception("向量库构建失败: %s", e)
        print(f"\n向量库构建失败: {e}")
        return False


def run_pipeline(
    steps: list[str] | None = None,
    full_rebuild: bool = False,
    dry_run: bool = False,
) -> None:
    """
    按顺序运行流水线步骤。

    Args:
        steps: 要执行的步骤列表，None 表示全部。可选值: ['clean', 'extract', 'build']
        full_rebuild: 是否强制全量重建向量库
        dry_run: 仅打印计划，不实际执行
    """
    all_steps = steps or ["clean", "extract", "build"]

    print("=" * 60)
    print("  海洋工程 RAG 知识库流水线")
    print("=" * 60)
    print(f"  模式: {'DRY-RUN (预演)' if dry_run else '执行'}")
    print(f"  步骤: {' -> '.join(all_steps)}")
    if full_rebuild:
        print(f"  向量库: 强制全量重建")
    print("=" * 60)

    if dry_run:
        # 打印每个步骤会做什么
        if "clean" in all_steps:
            print("\n[clean] 将扫描 data/input/reports/ 和 data/input/standards/ 中的 PDF/Office 文件")
            print("        使用 Docling + VLM 解析并输出 Markdown 到 data/cleaned/")
        if "extract" in all_steps:
            print("\n[extract] 将扫描 data/cleaned/reports/ 和 data/cleaned/standards/ 中的 .md 文件")
            print("         使用 LLM 提取元数据并输出同名 .json")
        if "build" in all_steps:
            strategy = "全量重建" if full_rebuild else "智能增量"
            print(f"\n[build] 将扫描 .md + .json 文件，切块嵌入存入 Chroma（{strategy}）")
        return

    start_time = time.time()

    step_funcs = {
        "clean": lambda: step_clean(),
        "extract": lambda: step_extract(),
        "build": lambda: step_build(full_rebuild=full_rebuild),
    }

    for step_name in all_steps:
        if step_name not in step_funcs:
            _log.error("未知步骤: %s", step_name)
            print(f"未知步骤: {step_name}，可选: clean, extract, build")
            sys.exit(1)

        ok = step_funcs[step_name]()
        if not ok:
            _log.error("步骤 '%s' 失败，流水线中止", step_name)
            print(f"\n步骤 '{step_name}' 失败，流水线中止。请修复后重试。")
            sys.exit(1)

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)

    if h > 0:
        time_str = f"{int(h)}小时 {int(m)}分钟 {s:.2f}秒"
    elif m > 0:
        time_str = f"{int(m)}分钟 {s:.2f}秒"
    else:
        time_str = f"{s:.2f}秒"

    print("\n" + "=" * 60)
    print(f"  流水线全部完成！总耗时: {time_str}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="海洋工程 RAG 知识库流水线 — 自动化构建全流程"
    )
    parser.add_argument(
        "--step",
        choices=["clean", "extract", "build"],
        help="仅运行指定步骤 (可多次使用 --step 组合)",
        action="append",
        dest="steps",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="强制全量重建向量库（跳过增量逻辑）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的操作，不实际执行",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            steps=args.steps,
            full_rebuild=args.full_rebuild,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        stop_server()
