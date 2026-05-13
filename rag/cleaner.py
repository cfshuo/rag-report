# -*- coding: utf-8 -*-
"""
cleaner.py — Docling + Local VLM (InternVL 30B) 自动化联合解析

集成 LM Studio 自动模型调度与 API 服务，遵循先转 PDF 再清洗原则。
针对复杂海洋工程文档：图表、公式、表格优先交由 30B 视觉大模型解析。
独家优化：将 Docling 底层表格寻边模型强制转移至 CPU 运行，彻底解决单卡 OOM 问题！
"""

import base64
import io
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
# 🛑 核心引入：引入 CPU 加速器控制模块
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import (
    FormulaItem,
    PictureItem,
    TableItem,
    TextItem,
)
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

import config
from rag.utils.hashing import compute_file_hash, load_hash_cache, save_hash_cache
from rag.utils.logging_config import setup_logging
from rag.utils.lms import load_model, unload_all, stop_server

_log = setup_logging("rag.cleaner")

# 输入目录 → 输出目录映射
_INPUT_DIRS = {
    config.ORIGIN_REPORT_DIR: config.CLEAN_REPORT_DIR,
    config.ORIGIN_STANDARD_DIR: config.CLEAN_STANDARD_DIR,
}

# 🛑 核心配置：表格重新纳入 VLM 处理范围，享受 30B 大模型的强力解析
_VLM_LABELS = {
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.CHART,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.FORMULA,
}

_HEADING_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.SECTION_HEADER,
}


# ==============================================================================
# Module 1: VLM Caller
# ==============================================================================
class VLMCaller:
    """视觉大模型调用器，将文档中的图片元素转换为 Markdown/LaTeX。"""

    def __init__(self, base_url: str = config.API_BASE_URL):
        self._client = OpenAI(base_url=base_url, api_key=config.API_KEY)
        self._model_name = config.VLM_MODEL_NAME
        _log.info("VLM 模型已连接: %s", self._model_name)

    @staticmethod
    def _encode_image(pil_image: Image.Image) -> str:
        """将 PIL 图片编码为 base64 字符串。"""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query(self, pil_image: Image.Image) -> str:
        """调用 VLM 识别图片内容。"""
        img_b64 = self._encode_image(pil_image)
        for attempt in range(1 + config.VLM_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": config.VLM_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    },
                                },
                                {"type": "text", "text": config.VLM_USER_TEXT},
                            ],
                        },
                    ],
                    temperature=0.1,  # 极低温度，保证工程数据提取的严谨性
                    timeout=config.VLM_TIMEOUT,
                    max_tokens=4096,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                _log.warning(
                    "VLM 调用失败 (尝试 %d/%d): %s",
                    attempt + 1, 1 + config.VLM_MAX_RETRIES, e,
                )
        return ""


# ==============================================================================
# Module 2: Document Processor & Converter
# ==============================================================================
def convert_office_to_pdf(input_path: Path) -> None:
    """使用 LibreOffice 将 Word 或 PPT 转换为 PDF。"""
    output_dir = input_path.parent
    expected_pdf_path = output_dir / f"{input_path.stem}.pdf"

    if expected_pdf_path.exists():
        _log.info("PDF 已存在，跳过转换: %s", expected_pdf_path.name)
        return

    _log.info("正在将 Office 文件转换为 PDF: %s", input_path.name)
    try:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(input_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _log.info("转换成功: %s", input_path.name)
    except subprocess.CalledProcessError as e:
        _log.error("Office 转 PDF 失败 [%s]: %s", input_path.name, e)


def _get_item_image(item, doc) -> Image.Image | None:
    """从 Docling 解析结果中提取图片元素。"""
    try:
        if hasattr(item, "get_image"):
            pil_img = item.get_image(doc)
            if pil_img is not None:
                return pil_img
    except Exception:
        pass
    try:
        img_ref = getattr(item, "image", None)
        if img_ref is not None:
            return img_ref.pil_image()
    except Exception:
        pass
    return None


def _table_to_markdown(table_item: TableItem, doc) -> str:
    """
    终极兜底方案：纯 Python 坐标网格拆分器。
    当 30B 模型罢工时，用这段代码强行拆分合并单元格，防数据丢失。
    """
    try:
        num_rows = table_item.data.num_rows
        num_cols = table_item.data.num_cols

        if num_rows == 0 or num_cols == 0:
            return ""

        grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]

        for cell in table_item.data.table_cells:
            text = cell.text.replace("\n", " ").strip() if cell.text else ""
            start_r = cell.start_row_offset_idx
            end_r = cell.end_row_offset_idx
            start_c = cell.start_col_offset_idx
            end_c = cell.end_col_offset_idx

            for r in range(start_r, end_r + 1):
                for c in range(start_c, end_c + 1):
                    if r < num_rows and c < num_cols:
                        grid[r][c] = text

        lines = []
        for r, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if r == 0:
                lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        return "\n".join(lines)

    except Exception as e:
        _log.warning(f"自定义表格降维解析失败，返回空: {e}")
        return ""


def _strip_vlm_code_fence(text: str) -> str:
    """去掉 VLM 多余的 ```markdown / ``` 包裹，保留纯 Markdown 内容。"""
    text = text.strip()
    fence_patterns = ["```markdown", "```md", "```latex", "```"]
    for fence in fence_patterns:
        if text.startswith(fence) and text.rstrip().endswith("```"):
            inner = text[len(fence):]
            if inner.endswith("```"):
                inner = inner[:-3]
            return inner.strip()
    return text


def _is_table_or_figure_caption(text: str) -> bool:
    """检测文本是否为表格/图片标题。"""
    import re as _re
    return bool(_re.match(r"[表图]\s*\d+(?:[\.-]\d+)*", text))


def _render_text_item(item, tree_depth: int) -> str:
    """将文本类 Docling 元素渲染为 Markdown。"""
    text = getattr(item, "text", "") or ""
    text = text.strip()
    if not text:
        return ""

    if item.label in _HEADING_LABELS:
        heading_level = getattr(item, "level", None)
        if heading_level is None:
            heading_level = min(tree_depth + 1, 6)
        else:
            heading_level = min(heading_level, 6)
        prefix = "#" * heading_level
        return f"{prefix} {text}"
    return text


def process_document(input_path: Path, vlm: VLMCaller | None) -> str:
    """使用 Docling (+ VLM) 解析单个文档为 Markdown。"""
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0

    # 🛡️ 终极护城河：开启表格结构识别以截图发给 VLM，但强制底层模型在 CPU 运行，保住 5090 显存！
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,  # 调用 CPU 核心计算，绝不抢占 GPU
        device=AcceleratorDevice.CPU
    )

    if vlm is not None:
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(input_path)
    doc = result.document
    output: list[str] = []
    pending: list[str] = []  # 暂存最近一个表格/图片的输出行，等待可能跟随的标题

    for item, level in doc.iterate_items():
        label = item.label

        if label in _VLM_LABELS:
            if pending:
                output.extend(pending)
                pending = []

            vlm_result = ""
            if vlm is not None:
                img = _get_item_image(item, doc)
                if img is not None:
                    vlm_result = vlm.query(img)

            if vlm_result:
                vlm_result = _strip_vlm_code_fence(vlm_result)
                if label == DocItemLabel.FORMULA and not (
                        vlm_result.startswith("$") or vlm_result.startswith("\\[")
                ):
                    pending = [f"$$\n{vlm_result}\n$$", ""]
                else:
                    pending = [vlm_result, ""]

            # 🛡️ 容灾兜底机制：如果 30B 巨兽超时失败，且当前是个表格，立刻呼叫 Python 引擎救场
            elif isinstance(item, TableItem):
                if vlm is not None:
                    _log.warning("⚠️ 30B VLM 处理表格失败或超时，降级调用 Python 坐标引擎强行提取。")
                fallback = _table_to_markdown(item, doc)
                if fallback:
                    pending = [fallback, ""]

            # 公式解析失败
            elif label == DocItemLabel.FORMULA:
                text = getattr(item, "text", "") or ""
                text = text.strip()
                if text:
                    if text.startswith("$$") or text.startswith("\\[") or text.startswith("$"):
                        pending = [text, ""]
                    else:
                        pending = [f"$$\n{text}\n$$", ""]
                else:
                    pending = ["\n> **公式元素解析失败**\n", ""]
            else:
                pending = [config.IMAGE_PLACEHOLDER, ""]

        else:
            rendered = _render_text_item(item, level)
            if rendered:
                if pending and _is_table_or_figure_caption(rendered):
                    # 标题倒装：标题放在表格/图片之前
                    pending = [rendered, ""] + pending
                    output.extend(pending)
                else:
                    if pending:
                        output.extend(pending)
                    output.append(rendered)
                    output.append("")
                pending = []

    if pending:
        output.extend(pending)

    return "\n".join(output)


# ==============================================================================
# Module 3: File Traversal & Main
# ==============================================================================
def main(use_vlm: bool, full_rebuild: bool = False) -> None:
    """文档清洗主流程。"""
    start_time = time.time()

    for input_dir, output_dir in _INPUT_DIRS.items():
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    office_files: list[Path] = []
    for input_dir in _INPUT_DIRS:
        if input_dir.exists():
            for file_path in input_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in config.OFFICE_SUFFIXES:
                    office_files.append(file_path)

    if office_files:
        print(f"\n发现 {len(office_files)} 个 Office 文件 (Word/PPT)，执行预处理转换...")
        for f in tqdm(office_files, desc="转换进度", unit="file"):
            convert_office_to_pdf(f)

    pdf_files: list[tuple[Path, Path]] = []
    for input_dir, output_dir in _INPUT_DIRS.items():
        if input_dir.exists():
            for file_path in input_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() == ".pdf":
                    pdf_files.append((file_path, output_dir))

    if not pdf_files:
        _log.info("未在目录中找到任何 PDF 文件可供清洗。")
        return

    old_hashes = {} if full_rebuild else load_hash_cache(config.CLEANER_CACHE_FILE)
    to_process: list[tuple[Path, Path]] = []
    skipped = 0
    for input_path, output_dir in pdf_files:
        output_path = output_dir / f"{input_path.stem}.md"
        current_hash = compute_file_hash(input_path)
        if not full_rebuild and output_path.exists() and current_hash:
            key = str(input_path)
            if key in old_hashes and old_hashes[key] == current_hash:
                skipped += 1
                continue
        to_process.append((input_path, output_dir))

    if skipped:
        print(f"\n增量模式：跳过 {skipped} 份未变更的 PDF，待清洗 {len(to_process)} 份。")
    else:
        print(f"\n格式化完毕，共准备清洗 {len(to_process)} 个 PDF 文件。")

    if not to_process:
        print("所有文档已是最新，无需清洗。")
        return

    new_hashes: dict[str, str] = {}
    vlm = VLMCaller() if use_vlm else None
    success = 0
    failed = 0

    for input_path, output_dir in tqdm(to_process, desc="清洗进度", unit="file"):
        tqdm.write(f"正在处理: {input_path.name}")

        try:
            markdown_content = process_document(input_path, vlm)
            output_path = output_dir / f"{input_path.stem}.md"
            output_path.write_text(markdown_content, encoding="utf-8")
            _log.info("成功输出: %s", output_path.name)
            file_hash = compute_file_hash(input_path)
            if file_hash:
                new_hashes[str(input_path)] = file_hash
            success += 1
        except Exception:
            _log.error("处理失败: %s\n%s", input_path.name, traceback.format_exc())
            failed += 1

    if not full_rebuild:
        for key, val in old_hashes.items():
            if key not in new_hashes:
                new_hashes[key] = val
    save_hash_cache(config.CLEANER_CACHE_FILE, new_hashes)

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)
    time_str = f"{int(h)}小时 {int(m)}分钟 {s:.2f}秒" if h > 0 else f"{int(m)}分钟 {s:.2f}秒" if m > 0 else f"{s:.2f}秒"

    print(f"\n清洗任务完成。成功: {success}, 失败: {failed}")
    print(f"总耗时: {time_str}")


if __name__ == "__main__":
    if config.USE_VLM:
        print("\n模式: [Docling (CPU寻边) + 视觉大模型 (GPU解析)] 联合解析模式已开启")
        try:
            load_model("vlm", gpu_ratio="max")
            main(use_vlm=True)
        except Exception as main_e:
            _log.exception("程序因严重错误中断: %s", main_e)
            print(f"\n程序因严重错误中断: {main_e}")
        finally:
            print("\n进入清理流程...")
            unload_all()
            stop_server()
    else:
        print("\n模式: [纯 Docling] 极速提取模式已开启 (已关闭视觉大模型)")
        main(use_vlm=False)