# -*- coding: utf-8 -*-
"""
文件哈希工具模块

用于增量向量数据库构建 —— 通过比对文件 SHA-256 哈希值，
判断文档是否在上次构建后发生了变化，避免重复嵌入计算。

Usage:
    from utils.hashing import compute_file_hash, get_changed_files
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple

_log = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """
    计算文件的 SHA-256 哈希值。

    Args:
        file_path: 文件路径

    Returns:
        64 位十六进制哈希字符串，失败时返回空字符串
    """
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError as e:
        _log.warning("无法计算文件哈希 %s: %s", file_path.name, e)
        return ""


def load_hash_cache(cache_path: Path) -> dict[str, str]:
    """
    加载哈希缓存文件。

    Args:
        cache_path: 缓存文件路径 (.json)

    Returns:
        {文件路径字符串: SHA256哈希}，文件不存在或损坏时返回空字典
    """
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _log.warning("哈希缓存文件损坏，将全量重建: %s", e)
        return {}


def save_hash_cache(cache_path: Path, hashes: dict[str, str]) -> None:
    """
    保存哈希缓存到 JSON 文件。

    Args:
        cache_path: 缓存文件路径
        hashes: {文件路径字符串: SHA256哈希}
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_changed_files(
    file_paths: list[Path],
    cache_path: Path,
) -> Tuple[list[Path], list[Path], list[Path]]:
    """
    比对文件列表与缓存，返回 (新增, 修改, 未变) 三类文件。

    增量策略:
      - 新增: 缓存中不存在的文件
      - 修改: 哈希值变更的文件
      - 未变: 哈希值相同的文件 (跳过嵌入)

    Args:
        file_paths: 当前所有待处理文件
        cache_path: 哈希缓存文件路径

    Returns:
        三元组 (new_files, modified_files, unchanged_files)
    """
    old_hashes = load_hash_cache(cache_path)

    new_files: list[Path] = []
    modified_files: list[Path] = []
    unchanged_files: list[Path] = []

    for fp in file_paths:
        key = str(fp)
        current_hash = compute_file_hash(fp)

        if not current_hash:
            # 无法计算哈希的当作新文件处理
            new_files.append(fp)
            continue

        if key not in old_hashes:
            new_files.append(fp)
        elif old_hashes[key] != current_hash:
            modified_files.append(fp)
        else:
            unchanged_files.append(fp)

    _log.info(
        "文件变更检测: 新增 %d, 修改 %d, 未变 %d",
        len(new_files), len(modified_files), len(unchanged_files),
    )
    return new_files, modified_files, unchanged_files
