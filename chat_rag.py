# -*- coding: utf-8 -*-
"""
chat_rag.py — 海洋工程 RAG 对话系统 (终端纯净输出版)

基于检索增强生成 (RAG) 的海洋水文气象专家问答引擎。
从本地 Chroma 向量库检索相关知识片段，结合大语言模型生成专业回答。
"""

import logging
import readline  # noqa: F401 — 强制激活终端行编辑（退格/方向键/历史）
import re
import warnings
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
    """从查询中提取条款编号。"""
    numbers: set[str] = set()
    for p in _CLAUSE_PATTERNS:
        numbers.update(p.findall(query))
    return numbers


def _hybrid_retrieval(db, query: str, top_k: int):
    """混合检索：向量语义搜索 + 条款编号定向向量检索（避免全库扫描）。"""
    clause_nums = _extract_clause_numbers(query)
    if not clause_nums:
        return db.similarity_search(query, k=top_k)

    # 条款编号定向向量检索 — 构造针对性的查询语句，走 HNSW 索引 O(log N)
    seen: set[str] = set()
    clause_hits: list = []
    for num in clause_nums:
        for doc in db.similarity_search(f"第{num}条 条款{num}", k=3):
            if doc.page_content not in seen:
                clause_hits.append(doc)
                seen.add(doc.page_content)

    _log.info("条款编号 %s 向量命中 %d 个 chunk", clause_nums, len(clause_hits))

    # 主语义检索补充
    for doc in db.similarity_search(query, k=top_k):
        if doc.page_content not in seen:
            clause_hits.append(doc)
            seen.add(doc.page_content)

    return clause_hits[:top_k]


def _clean_display_name(name: str) -> str:
    """
    【核心修复】彻底清除字符串中的所有空白字符（空格、换行等）。
    这解决了参考来源中出现额外空格的问题，同时也用于去重。
    """
    return re.sub(r'\s+', '', name)


def _build_context(docs) -> tuple[str, Set[str]]:
    """从检索文档构建结构化的上下文文本和来源集合。"""
    context_text = ""
    source_tracker = {}

    for i, doc in enumerate(docs):
        meta = doc.metadata
        doc_type = meta.get("文档类型", "")
        fname = meta.get("来源文件", "未知文件")

        context_text += f"\n【参考资料 {i + 1}】"

        if doc_type == "规范":
            # 使用清洗后的名称，确保没有多余空格
            raw_std_name = meta.get("标准名称", "") or _extract_std_name_from_filename(fname)
            std_name = _clean_display_name(raw_std_name)
            std_code = _clean_display_name(meta.get("标准编号", ""))

            context_text += f"\n- 文档类型: 规范"
            if std_code:
                context_text += f"\n- 标准编号: {std_code}"
            context_text += f"\n- 标准名称: {std_name}"

            ref_display = f"《{std_name}》"
            if std_code:
                ref_display += f" ({std_code})"

            # 使用清洗后的 key 进行去重追踪
            source_tracker[ref_display] = ref_display

        else:
            proj = meta.get("项目名称", "未知项目")
            year = meta.get("编制年份", "未知年份")
            loc = meta.get("海域位置", "未知位置")
            stage = meta.get("设计阶段", "未知阶段")
            context_text += f"\n- 项目名称: {proj}"
            context_text += f"\n- 编制年份: {year}"
            context_text += f"\n- 海域位置: {loc}"
            context_text += f"\n- 设计阶段: {stage}"

            ref_display = f"{proj}, {year}, {loc}, {stage}"
            source_tracker[ref_display] = ref_display

        context_text += f"\n- 来源文件: {fname}"
        context_text += f"\n[正文片段内容:\n{doc.page_content}\n"
        context_text += "-" * 30 + "\n"

    return context_text, set(source_tracker.values())


def _extract_std_name_from_filename(filename: str) -> str:
    """从文件名中提取规范名称，如 '《XXX》出版稿2019.md' → 'XXX'。"""
    m = re.search(r"《(.+?)》", filename)
    return m.group(1) if m else filename


# 上标/下标数字 → Unicode 字符映射
_SUP_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
    'n': 'ⁿ',
}
_SUB_MAP = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
}


def _convert_sup_sub(text: str) -> str:
    """将 LaTeX 风格的 ^2 ^{2} _2 _{2} 转为 Unicode 上标/下标。"""
    # ^{...} 上标
    text = re.sub(r'\^\{([^}]+)\}', lambda m: ''.join(_SUP_MAP.get(c, c) for c in m.group(1)), text)
    # ^单个数字
    text = re.sub(r'\^(\d)', lambda m: _SUP_MAP.get(m.group(1), m.group(1)), text)
    # _{...} 下标
    text = re.sub(r'\_\{([^}]+)\}', lambda m: ''.join(_SUB_MAP.get(c, c) for c in m.group(1)), text)
    # _单个数字
    text = re.sub(r'_(\d)', lambda m: _SUB_MAP.get(m.group(1), m.group(1)), text)
    return text


def _strip_latex_commands(text: str) -> str:
    """去掉 LaTeX 反斜杠命令，转换为纯文本符号，保留 {} 和普通符号。"""
    latex_map = {
        r'\ge': '>=', r'\le': '<=', r'\geq': '>=', r'\leq': '<=',
        r'\ne': '!=', r'\neq': '!=',
        r'\pm': '+-', r'\mp': '-+', r'\times': 'x', r'\cdot': '*',
        r'\approx': '~=', r'\sim': '~', r'\propto': '正比于',
        r'\infty': '无穷大', r'\to': '->', r'\rightarrow': '->',
        r'\leftarrow': '<-', r'\Rightarrow': '=>', r'\Leftrightarrow': '<=>',
        r'\text': '', r'\mathrm': '', r'\mathbf': '', r'\mathit': '',
        r'\textsuperscript': '', r'\textsubscript': '',
        r'\degree': '℃', r'\deg': '℃', r'\percent': '%',
        r'\cm': 'cm', r'\mm': 'mm', r'\km': 'km', r'\m': 'm',
        r'\kg': 'kg', r'\g': 'g', r'\s': 's', r'\min': 'min',
        r'\hour': 'hour', r'\ms': 'm/s', r'\cms': 'cm/s',
        r'\frac': '', r'\sqrt': 'sqrt', r'\sum': '求和', r'\prod': '求积',
        r'\int': '积分', r'\partial': '偏', r'\nabla': '梯度',
        r'\alpha': 'alpha', r'\beta': 'beta', r'\gamma': 'gamma',
        r'\delta': 'delta', r'\theta': 'theta', r'\pi': 'pi',
        r'\mu': 'mu', r'\sigma': 'sigma', r'\rho': 'rho',
        r'\omega': 'omega', r'\lambda': 'lambda',
    }
    for latex, plain in latex_map.items():
        text = text.replace(latex, plain)
    # 上标/下标转换：必须在去掉反斜杠命令之后做，避免破坏 LaTeX 命令
    text = _convert_sup_sub(text)
    # 去掉残留的反斜杠命令
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text


def _clean_model_output(text: str) -> str:
    """后处理：清除 LaTeX 命令和 Markdown 修饰符，保留 {}、$、* 等普通符号。"""
    # 1. 去掉 Markdown 加粗/斜体修饰符，但保留单独的 * 符号（乘号等）
    text = re.sub(r'\*{2,}([^*]+?)\*{2,}', r'\1', text)   # **bold** (2个及以上*)
    text = re.sub(r'(?<!\*)\*([^*\s][^*]*?[^*\s])\*(?!\*)', r'\1', text)  # *italic* (单个*包裹)
    text = re.sub(r'_{2,}([^_]+?)_{2,}', r'\1', text)     # __underline__
    # 2. 去掉 Markdown 标题标记（行首 #），保留正文中的 # 号
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 3. 去掉行内 LaTeX $...$ 和 \(...\)（仅去掉 $ 和 \( \) 定界符，保留内部内容）
    text = re.sub(r'\$([^$]+?)\$', lambda m: _strip_latex_commands(m.group(1)), text)
    text = re.sub(r'\\\(([^)]+?)\\\)', lambda m: _strip_latex_commands(m.group(1)), text)
    # 4. 去掉独立公式 $$...$$ 和 \[...\] 块
    text = re.sub(r'\$\$[\s\S]*?\$\$', lambda m: '[公式: ' + _strip_latex_commands(m.group(0)[2:-2].strip()) + ']', text)
    text = re.sub(r'\\\[[\s\S]*?\\\]', lambda m: '[公式: ' + _strip_latex_commands(m.group(0)[2:-2].strip()) + ']', text)
    # 5. 去掉残留的 LaTeX 命令
    text = _strip_latex_commands(text)
    # 6. 清理多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _build_system_prompt(context_text: str) -> str:
    """
    构建系统提示词。
    解除对模型输出格式的死板限制，让 9B 模型发挥最自然的语言能力，
    后续的终端纯净化交由 Python 的 _clean_model_output 函数处理。
    """
    # 预处理：清洗参考资料中的 LaTeX/Markdown 格式，防止输入端污染
    clean_context = _clean_model_output(context_text)

    # 上下文截断保护
    max_chars = max(1000, int((config.LLM_CONTEXT_LENGTH - 1500) * 0.6))
    if len(clean_context) > max_chars:
        clean_context = clean_context[:max_chars] + "\n...[上下文已截断]"
        _log.warning("上下文过长 (%d 字符)，已截断至 %d 字符", len(context_text), max_chars)

    return (
        "你是一位专业的海洋水文气象高级工程师。\n"
        "请严谨地基于提供的【参考资料】来回答用户的问题。\n\n"
        "【回答纪律】：\n"
        "1. 请逻辑清晰、分点作答。如果涉及数学关系，尽量使用通俗易懂的文本描述。\n"
        "2. 如果资料中没有相关信息，请直接回答：“根据提供的资料，无法回答该问题。”，绝对不要编造数据或凭空猜测。\n"
        "3. 禁止使用任何 Markdown 格式（不要使用 **、*、#、```、___ 等符号），输出纯文本即可。\n\n"
        f"以下是相关参考资料：\n{clean_context}"
    )


def chat_loop() -> None:
    """RAG 对话主循环。"""
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
    print("🌊 海洋工程智能专家库 (RAG 系统) 已准备就绪")
    print("输入 'exit' 或 'quit' 退出对话")
    print("=" * 60)

    # 【核心修复】移除了 chat_history，实现"一问一答"无状态模式
    # 这样可以避免上下文累积导致的显存溢出或逻辑混乱

    while True:
        try:
            user_query = _safe_input("\n🧑‍💻 工程师提问: ")
        except (EOFError, KeyboardInterrupt):
            print("\n用户退出。")
            break

        if user_query.lower() in ("exit", "quit", "退出"):
            break
        if not user_query.strip():
            continue

        print("🔍 正在检索相关水文报告与规范片段...")
        docs = _hybrid_retrieval(db, user_query, config.RETRIEVAL_K)

        if not docs:
            _log.info("未检索到相关知识: %s", user_query[:80])
            print("🤖 专家回答: \n系统未在库中找到相关知识。")
            continue

        context_text, source_files = _build_context(docs)
        system_prompt = _build_system_prompt(context_text)

        # 构建仅包含当前问题和系统提示的消息列表
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        print("🤖 专家回答: \n", end="", flush=True)

        try:
            resp = _client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                stream=False,
                timeout=120.0,
            )

            full_response = resp.choices[0].message.content or ""

            if not full_response.strip():
                _log.warning("模型返回空内容 — query=%.80s", user_query)
                print("[系统提示：模型未返回有效内容]")
            else:
                # 终端纯净输出：去除可能的 markdown 残留后直接打印
                clean_text = _clean_model_output(full_response)
                print(clean_text)
                print("\n📑 [参考来源]:")
                for s in sorted(source_files):
                    print(f"- {s}")

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