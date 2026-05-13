# -*- coding: utf-8 -*-
"""
项目全局配置 — 海洋工程水文气象 RAG 知识库系统

所有路径、模型名称、业务参数均在此集中管理。
其他模块统一通过 `import config` 引用，避免分散硬编码。
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Literal

# ==========================================
# 1. 基础路径配置
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

# 统一数据根目录
DATA_DIR = BASE_DIR / "data"

# 原始输入文档目录
ORIGIN_REPORT_DIR = DATA_DIR / "input" / "reports"
ORIGIN_STANDARD_DIR = DATA_DIR / "input" / "standards"

# 清洗后输出目录
CLEAN_REPORT_DIR = DATA_DIR / "cleaned" / "reports"
CLEAN_STANDARD_DIR = DATA_DIR / "cleaned" / "standards"

# 向量数据库存储目录
CHROMA_DB_DIR = str(DATA_DIR / "chroma")

# 日志目录
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 增量嵌入缓存文件 (记录已处理文件的哈希值)
EMBEDDING_CACHE_FILE = DATA_DIR / "chroma" / ".file_hashes.json"

# 增量清洗缓存 (PDF → MD 的哈希映射)
CLEANER_CACHE_FILE = DATA_DIR / "cleaned" / ".cleaner_hashes.json"

# 增量提取缓存 (MD → JSON 的哈希映射)
EXTRACTOR_CACHE_FILE = DATA_DIR / "cleaned" / ".extractor_hashes.json"

# ==========================================
# 2. 模型与 API 配置
# ==========================================
API_BASE_URL = "http://localhost:1234/v1"
API_KEY = "local-llm"

# 视觉大模型 (用于 document_cleaner 的表格/公式识别)
VLM_MODEL_NAME = "opengvlab_internvl3_5-8b"

# 向量嵌入模型 (用于 vector_database_builder 和 chat_rag)
EMBEDDING_MODEL_NAME = "text-embedding-bge-m3"

# 逻辑推理大模型 (用于 information_extract 和 chat_rag)
# 备选: "qwen/qwen3.5-35b-a3b" (35B MoE, 22GB, 质量高但慢)
LLM_MODEL_NAME = "qwen/qwen3.5-9b"

# GPU 显存分配比例
GPU_OFFLOAD: Dict[str, str] = {
    "vlm":       "max",   # InternVL 视觉模型
    "embedding": "max",   # bge-m3 向量模型 (634MB, 全上 GPU)
    "llm":       "max",   # Qwen 9B 推理模型 (6.5GB, 轻松全上 GPU)
}

# 推理上下文长度
LLM_CONTEXT_LENGTH = 32768
EMBEDDING_CONTEXT_LENGTH = 4096

# VLM 调用配置
USE_VLM = True
VLM_TIMEOUT = 120
VLM_MAX_RETRIES = 2

# ==========================================
# 3. 文档切块配置
# ==========================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Markdown 标题切分层级
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

# ==========================================
# 4. RAG 检索配置
# ==========================================
RETRIEVAL_K = 3

# ==========================================
# 5. 元数据提取配置
# ==========================================
# 报告类文档的提取字段
METADATA_FIELDS_REPORT = {
    "项目名称": "示例项目名称",
    "海域位置": "示例海域位置",
    "设计阶段": "示例阶段",
    "编制年份": "2024",
}

# 标准类文档的提取字段
METADATA_FIELDS_STANDARD = {
    "标准编号": "GB/T 123",
    "标准名称": "示例标准名称",
    "规范类别": "国家标准",
    "发布年份": "2024",
}

# 用于元数据提取的最大字符数 (None = 读取全文)
MAX_TEXT_FOR_METADATA: int | None = None

# ==========================================
# 6. Office 文件转换配置
# ==========================================
OFFICE_SUFFIXES = {".doc", ".docx", ".ppt", ".pptx"}

# ==========================================
# 7. VLM 提示词与标签配置
# ==========================================
VLM_SYSTEM_PROMPT = (
    "你是一个严谨的工程文档数据提取专家。请将图片中的表格精准转换为 Markdown 表格，"
    "或将图片中的数学公式转换为 LaTeX 格式（行内使用 $...$，独立公式使用 $$...$$）。"
    "只输出 Markdown 或 LaTeX 代码，不要解释。"
    "在解析表格并转换为 Markdown 格式时，由于 Markdown 原生不支持单元格的跨行(rowspan)或跨列(colspan)合并，你必须遵守以下铁律：对于表格中的合并单元格，请务必将其文本内容【完整复制并拆分填充】到每一个对应的 Markdown 独立单元格中！严禁在合并对应的下方或右侧单元格中留空。确保每一行的数据独立且完整！"
)
VLM_USER_TEXT = "请提取这张图片中的核心信息并转化为代码格式："
IMAGE_PLACEHOLDER = ""

# ==========================================
# 8. 配置校验
# ==========================================
def validate_config() -> Dict[str, List[str]]:
    """
    校验配置完整性，返回 {级别: [消息列表]}。
    无问题时返回空字典。

    Returns:
        dict: 例如 {"warning": ["...", "..."], "error": ["..."]}
    """
    issues: Dict[str, List[str]] = {"error": [], "warning": []}

    # 必要目录
    for name, path in [
        ("ORIGIN_REPORT_DIR", ORIGIN_REPORT_DIR),
        ("ORIGIN_STANDARD_DIR", ORIGIN_STANDARD_DIR),
    ]:
        if not path.exists():
            issues["warning"].append(f"{name} 不存在: {path}")

    # 输出目录可创建
    for name, path in [
        ("CLEAN_REPORT_DIR", CLEAN_REPORT_DIR),
        ("CLEAN_STANDARD_DIR", CLEAN_STANDARD_DIR),
    ]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            issues["error"].append(f"无法创建 {name}: {path} — {e}")

    # 模型名称
    for name in ["LLM_MODEL_NAME", "EMBEDDING_MODEL_NAME", "VLM_MODEL_NAME"]:
        if not globals().get(name):
            issues["error"].append(f"{name} 未配置")

    # API 地址
    if not API_BASE_URL:
        issues["error"].append("API_BASE_URL 未配置")

    # 切块参数
    if CHUNK_SIZE <= CHUNK_OVERLAP:
        issues["error"].append(
            f"CHUNK_SIZE ({CHUNK_SIZE}) 必须大于 CHUNK_OVERLAP ({CHUNK_OVERLAP})"
        )

    return {k: v for k, v in issues.items() if v}


_log = logging.getLogger(__name__)
