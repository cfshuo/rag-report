# -*- coding: utf-8 -*-
"""
chat_rag.py — 海洋工程 RAG 对话系统 (终端纯净输出版)

基于检索增强生成 (RAG) 的海洋水文气象专家问答引擎。
从本地 Chroma 向量库检索相关知识片段，结合大语言模型生成专业回答。
"""

import logging
import readline  # noqa: F401 — 强制激活终端行编辑（退格/方向键/历史）
import re
import time
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

    # 条款编号定向向量检索 — 将编号与原始查询语义融合，走 HNSW 索引 O(log N)
    seen: set[str] = set()
    clause_hits: list = []
    # 去除条款编号、标准代号等泛化词，提取纯语义关键词做定向检索
    semantic_query = re.sub(r'第\s*\d+(?:\.\d+)*\s*[条章节款]', '', query).strip()
    semantic_query = re.sub(r'[条款章节]\s*\d+(?:\.\d+)*', '', semantic_query).strip()
    # 去掉标准编号 (如 NB/T31029-2019)，这类词会干扰条款级语义检索
    semantic_query = re.sub(r'[A-Z]{2,}/[A-Z]\s*\d+[-–—]\d+', '', semantic_query).strip()
    if not semantic_query:
        semantic_query = query
    for num in clause_nums:
        # 用条款号 + 语义查询融合，避免纯数字匹配失效
        clause_query = f"{num} {semantic_query}"
        for doc in db.similarity_search(clause_query, k=3):
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
    输出铁律：三段式结构（引言 → 核心解答 → 专家补充），参考文献由系统自动追加。
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
        "【输出格式要求 — 必须严格遵循以下三段式结构，违反即为不合格】：\n\n"
        "=== 第一段：引言与背景 ===\n"
        "用一句话平滑引入，必须包含「根据《XXX规范》第X条...」或「依据XXX资料...」，并简述问题背景。\n"
        "这段让读者知道答案来自哪里、讨论的是什么话题。\n\n"
        "=== 第二段：核心解答 ===\n"
        "根据问题类型，分两种情形处理：\n"
        "  - 多维度/复杂条目：使用「1、2、3」（阿拉伯数字）逐条列出，每条带清晰小标题。\n"
        "    示例：「1. 远海风电场：场区离海缆路由登陆点所在岸线最近距离大于65km的风电场。」\n"
        "  - 单一事实问题（查数值、查定义）：用一段话直接给出精准答案，严禁强行分条。\n"
        "    对单一事实使用「第一」「第二」「第三」「第四」属于严重违规的凑字数行为。\n\n"
        "=== 第三段：专家补充 / 发散思考 ===\n"
        "以「注：」或「补充说明：」开头，基于当前问题给出有价值的专业关联信息，例如：\n"
        "  - 相关概念的对照对比（问深海，可补充浅海的界限作为对照；问远海，可补充近海的定义）\n"
        "  - 实际工程中的注意事项\n"
        "  - 与问题相关的上下游知识\n"
        "此段展现你的专家视野，但须与问题密切相关，不得离题。\n\n"
        "---\n"
        "【正确示例 — 问「深远海的划分」】\n"
        "根据《海上风电场工程风能资源测量及海洋水文观测规范》（NB/T31029-2019），海上风电场根据离岸距离或场址水深大小进行划分，关于远海与深海的划分标准如下：\n"
        "1. 远海风电场：场区离海缆路由登陆点所在岸线最近距离大于65km的风电场。\n"
        "2. 深海风电场：场区水深大于理论最低潮位以下50m的风电场。\n"
        "注：规范中同时明确了近海风电场（离岸距离大于10km且不大于65km）与浅海风电场（水深在理论最低潮位以下0m~50m）的界限，作为深远海划分的对照依据。\n\n"
        "【错误示例 — 问「流速V≥100cm/s时流向准确度是多少」（单一事实）】\n"
        "「第一，依据规范... 第二，表格显示... 第三，当流速V≥100cm/s时，流向准确度为±5°。第四，因此准确度是±5°。」\n"
        "以上错误原因：对单一事实强行分条，第三点和第四点重复相同答案，充满废话。\n"
        "正确做法：引言一句话带出规范来源，第二段一句话给出答案，第三段补充说明，干净利落。\n\n"
        "【补充铁律】\n"
        "1. 输出纯文本，不要使用 Markdown 格式（不要用 **、*、#、``` 等符号）。\n"
        "2. 禁止描述参考资料的结构（如「该表格详细列出了...」是废话）。\n"
        "3. 禁止在结尾重复总结（如「因此，针对您询问的...」「综上所述...」）。答案只说一遍。\n"
        "4. 如果资料中没有相关信息，请直接回答：「根据提供的资料，无法回答该问题。」，绝对不要编造数据。\n"
        "5. 禁止在末尾单独输出参考文献名称或编号。参考文献由系统自动追加，你不需要画蛇添足。\n\n"
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
        t0 = time.time()
        docs = _hybrid_retrieval(db, user_query, config.RETRIEVAL_K)
        t1 = time.time()

        if not docs:
            _log.info("未检索到相关知识: %s", user_query[:80])
            print("🤖 专家回答: \n系统未在库中找到相关知识。")
            continue

        context_text, source_files = _build_context(docs)
        t2 = time.time()
        system_prompt = _build_system_prompt(context_text)
        t3 = time.time()

        # 构建仅包含当前问题和系统提示的消息列表
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        print("🤖 专家回答: \n", end="", flush=True)

        try:
            t4 = time.time()
            stream = _client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                stream=True,
                timeout=120.0,
            )

            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    print(delta.content, end="", flush=True)
            t5 = time.time()

            if not full_response.strip():
                _log.warning("模型返回空内容 — query=%.80s", user_query)
                print("[系统提示：模型未返回有效内容]")
            else:
                print("\n\n📑 [参考来源]:")
                for s in sorted(source_files):
                    print(f"- {s}")

                # 耗时打点（仅日志，不在终端输出）
                _log.info("耗时: 总%.1fs 检索%.1fs LLM推理%.1fs 提示词%d字",
                          t5 - t0, t1 - t0, t5 - t4, len(system_prompt) + len(user_query))
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