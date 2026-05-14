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
VLM_MODEL_NAME = "opengvlab_internvl3_5-30b-a3b"

# 向量嵌入模型 (用于 vector_database_builder 和 chat_rag)
EMBEDDING_MODEL_NAME = "text-embedding-bge-m3"

# 逻辑推理大模型 (用于 information_extract 和 chat_rag)
LLM_MODEL_NAME = "qwen/qwen3.5-9b"

# GPU 显存分配比例
GPU_OFFLOAD: Dict[str, str] = {
    "vlm":       "max",   # InternVL 30B 离线清洗，独占显存
    "embedding": "max",   # bge-m3 向量模型 (634MB)
    "llm":       "max",   # Qwen 3 8B 推理模型
}

# 推理上下文长度 (输入+输出总窗口，8192 兼顾 RAG 场景的响应速度)
LLM_CONTEXT_LENGTH = 16384
# 单次响应最大生成 token 数
MAX_OUTPUT_TOKENS = 4096
# 嵌入模型上下文窗口
EMBEDDING_CONTEXT_LENGTH = 4096

# VLM 调用配置
USE_VLM = True
VLM_TIMEOUT = 240  # 核心修改：30B 处理表格极慢，必须给足时间防止中断
VLM_MAX_RETRIES = 2

# ==========================================
# 3. 文档切块配置 (为 9B 模型专门扩容)
# ==========================================
CHUNK_SIZE = 800       # 增大尺寸，保证规范段落完整
CHUNK_OVERLAP = 150    # 增大重叠，防止关键数据被腰斩

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
RETRIEVAL_K = 5        # 增加召回数量，给 9B 提供更足的弹药

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

# 核心修改：最多只读前 4000 字提取标签，防止整本书塞爆显存
MAX_TEXT_FOR_METADATA = 4000

# ==========================================
# 6. Office 文件转换配置
# ==========================================
OFFICE_SUFFIXES = {".doc", ".docx", ".ppt", ".pptx"}

# ==========================================
# 7. VLM 提示词与标签配置 (30B 满血表格提取版)
# ==========================================
VLM_SYSTEM_PROMPT = (
    "你是一位资深的海洋工程图像与数据解析专家。请精准解析输入图片中的核心信息：\n"
    "1. 如果是数学公式：请转换为标准的 LaTeX 格式（行内使用 $...$，独立公式使用 $$...$$）。\n"
    "2. 如果是工程图表（风玫瑰图、曲线图、流程图等）：请用严谨的工程师语言详细描述数据趋势或转换为 Markdown 列表。\n"
    "3. 如果是数据表格：请将其精准转换为 Markdown 表格。\n"
    "【⚠️ 表格提取绝对铁律 - 关乎系统生死】：\n"
    "由于 Markdown 语法根本不支持跨行(rowspan)或跨列(colspan)的单元格合并，你必须在脑海中将表格完全拆解为规则的网格！\n"
    "1. 只要遇到合并单元格，你必须将该单元格的文本内容【完整地重复打字】填充到对应的每一个 Markdown 独立单元格中！\n"
    "2. 绝对不允许出现空的单元格（除非原图就是空的）。\n"
    "3. 绝对不允许把上一行的合并项在下一行留空！\n"
    "只输出最终的 Markdown 或 LaTeX 内容，绝对不要输出任何多余的解释、寒暄或代码块标记（如 ```markdown）。"
)
VLM_USER_TEXT = "请精准提取这张图片中的核心工程信息并转化为代码或结构化文本："
IMAGE_PLACEHOLDER = ""

# ==========================================
# 8. 配置校验
# ==========================================
def validate_config() -> Dict[str, List[str]]:
    """
    校验配置完整性，返回 {级别: [消息列表]}。
    无问题时返回空字典。
    """
    issues: Dict[str, List[str]] = {"error": [], "warning": []}

    for name, path in [
        ("ORIGIN_REPORT_DIR", ORIGIN_REPORT_DIR),
        ("ORIGIN_STANDARD_DIR", ORIGIN_STANDARD_DIR),
    ]:
        if not path.exists():
            issues["warning"].append(f"{name} 不存在: {path}")

    for name, path in [
        ("CLEAN_REPORT_DIR", CLEAN_REPORT_DIR),
        ("CLEAN_STANDARD_DIR", CLEAN_STANDARD_DIR),
    ]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            issues["error"].append(f"无法创建 {name}: {path} — {e}")

    for name in ["LLM_MODEL_NAME", "EMBEDDING_MODEL_NAME", "VLM_MODEL_NAME"]:
        if not globals().get(name):
            issues["error"].append(f"{name} 未配置")

    if not API_BASE_URL:
        issues["error"].append("API_BASE_URL 未配置")

    if CHUNK_SIZE <= CHUNK_OVERLAP:
        issues["error"].append(
            f"CHUNK_SIZE ({CHUNK_SIZE}) 必须大于 CHUNK_OVERLAP ({CHUNK_OVERLAP})"
        )

    return {k: v for k, v in issues.items() if v}


_log = logging.getLogger(__name__)