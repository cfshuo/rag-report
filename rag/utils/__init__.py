# -*- coding: utf-8 -*-
"""
utils — 海洋工程 RAG 项目公共工具包

提供 LM Studio 自动化控制、统一日志配置、文件哈希工具。
"""

from rag.utils.lms import (
    load_both_models,
    load_model,
    start_server,
    stop_server,
    unload_all,
    unload_models,
)
from rag.utils.logging_config import setup_logging
from rag.utils.hashing import (
    compute_file_hash,
    get_changed_files,
    load_hash_cache,
    save_hash_cache,
)

__all__ = [
    # LMS 控制
    "load_both_models",
    "load_model",
    "start_server",
    "stop_server",
    "unload_all",
    "unload_models",
    # 日志
    "setup_logging",
    # 哈希
    "compute_file_hash",
    "get_changed_files",
    "load_hash_cache",
    "save_hash_cache",
]
