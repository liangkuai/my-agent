"""
Memory 模块 —— 持久化记忆的完整生命周期管理。

提供五个核心能力：
1. 记忆存取    —— 以 Markdown + YAML frontmatter 格式读写单条记忆文件
2. 索引维护    —— 自动维护 MEMORY.md 索引文件，供 system prompt 注入
3. 智能提取    —— 从对话中由 LLM 自动识别用户偏好、项目事实等新记忆
4. 合并去重    —— 当记忆数量超过阈值时调用 LLM 合并重复、清理过时内容
5. 相关检索    —— 根据当前对话自动筛选相关记忆注入上下文

记忆文件格式（每条记忆一个 .md 文件，YAML frontmatter + Markdown 正文）：
    ---
    name: short-kebab-case-id
    description: one-line summary for index lookup
    type: user | feedback | project | reference
    ---
    Markdown 正文（支持 [[wikilink]] 引用其他记忆）

调用时机（均由 app.agent_loop 驱动，无需外部手动触发）：
- extract_memories：每次 agent_loop 结束（模型给出最终回复后）
- consolidate_memories：同上，仅当记忆数 >= CONSOLIDATE_THRESHOLD 时执行
- load_memories：每次 agent_loop 开始，向当前对话注入相关记忆
"""

import re
import json
import time
from pathlib import Path

from constant import MEMORY_DIR, MEMORY_INDEX, MODEL, CONSOLIDATE_THRESHOLD
from llm_client import client
import tools


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Markdown 文件中的 YAML frontmatter。

    约定格式：
        ---
        name: my-memory
        description: 一句话描述
        type: user
        ---
        正文内容...

    Args:
        text: 文件的完整原始文本。

    Returns:
        (meta, body) 元组：
        - meta: frontmatter 中解析出的键值对字典，键名已去除首尾空白
        - body: frontmatter 之后（第二个 "---" 之后）的正文，已去除首尾空白
        若文件不以 "---" 开头或解析失败，返回 ({}, text)。
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


def list_memory_files() -> list[dict]:
    """列出 MEMORY_DIR 下所有记忆文件（排除 MEMORY.md 索引文件本身）。

    Returns:
        list[dict]，每个元素包含：
        - filename: 文件名（如 "user-preference-tabs.md"）
        - name: 记忆标识（取自 frontmatter 的 name 字段，缺失时回退到 stem）
        - description: 一句话描述（取自 frontmatter 的 description 字段）
        - type: 记忆类型（user / feedback / project / reference）
        - body: 正文内容（frontmatter 之后的 Markdown 文本）
    """
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        result.append(
            {
                "filename": f.name,
                "name": meta.get("name", f.stem),
                "description": meta.get("description", ""),
                "type": meta.get("type", "user"),
                "body": body,
            }
        )
    return result


def write_memory_file(name: str, mem_type: str, description: str, body: str) -> Path:
    """创建或覆写一条记忆文件，写入后自动重建索引。

    文件名由 name 派生：转为小写、空格和斜杠替换为连字符，加 .md 后缀。
    例如 name="User Preference Tabs" → "user-preference-tabs.md"。

    Args:
        name: 记忆标识（将转为 kebab-case 文件名）。
        mem_type: 类型标签（user / feedback / project / reference）。
        description: 一行摘要，用于 MEMORY.md 索引中的描述列。
        body: Markdown 正文内容。

    Returns:
        写入的文件 Path 对象。
    """
    slug = name.lower().replace(" ", "-").replace("/", "-")
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    filepath.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
    )
    _rebuild_index()
    return filepath


def _rebuild_index() -> None:
    """扫描 MEMORY_DIR 重建 MEMORY.md 索引文件。

    索引格式为 Markdown 列表，每行一条：
        - [name](filename.md) — description

    索引内容由 context.update_session_context() → memory.read_memory_index()
    读取后注入 system prompt，让模型知道有哪些可用记忆。
    MEMORY.md 本身不在扫描范围内，避免自引用。
    """
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) — {desc}")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")


def extract_memories(messages: list) -> None:
    """从最近 10 条对话中由 LLM 自动提取新的记忆。

    调用时机：agent_loop 每次给出最终回复后（stop_reason != "tool_use"）。

    工作流程：
    1. 取最近 10 条消息，拼接为 role: content 格式的纯文本对话
    2. 列出已存在的所有记忆（名称 + 描述），避免重复提取
    3. 构造 prompt 让 LLM 判断是否有新的用户偏好、项目事实或反馈
    4. 解析 LLM 返回的 JSON 数组，逐条调用 write_memory_file 写入
    5. 打印彩色提示告知用户新增了多少条记忆

    容错设计：
    - 空对话直接返回，不消耗 API 配额
    - LLM 返回空数组 → 无新记忆，正常跳过
    - JSON 解析失败或 API 异常 → 静默跳过（记忆提取不应阻塞主流程）

    Args:
        messages: 完整的对话历史列表，每项为 {"role": ..., "content": ...} 格式。
    """
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(getattr(b, "text", ""))
                for b in content
                if getattr(b, "type", None) == "text"
            )
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)

    if not dialogue.strip():
        return

    existing = list_memory_files()
    existing_desc = (
        "\n".join(f"- {m['name']}: {m['description']}" for m in existing)
        if existing
        else "(none)"
    )

    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=800
        )
        text = tools.extract_text(response.content).strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass


def consolidate_memories() -> None:
    """当记忆文件数超过 CONSOLIDATE_THRESHOLD 时，调用 LLM 合并去重。

    调用时机：与 extract_memories 相同（agent_loop 每次最终回复后），
    但仅在 len(files) >= CONSOLIDATE_THRESHOLD 时执行。

    工作流程：
    1. 收集所有记忆文件的 name + description + body，拼接为 catalog 文本
    2. 构造 prompt 让 LLM 执行四步操作：
       - 合并重复条目
       - 删除过时或被后续记忆覆盖的条目
       - 控制总数在 30 条以内
       - 优先保留用户偏好类记忆
    3. 清空 MEMORY_DIR 下所有 .md 文件（保留 MEMORY.md）
    4. 逐条写入 LLM 返回的合并后记忆
    5. 打印彩色提示告知合并前后数量变化

    容错设计：LLM 调用失败或 JSON 解析异常时静默返回，清空操作不会执行
    （清空位于 try 块内，仅在 LLM 成功返回合法结果后才触发）。
    但注意：若 LLM 返回了合法 JSON 数组但内容为空，已有记忆仍会被清空——
    这是信任 LLM 判断的权衡。
    """
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return

    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )

    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3000
        )
        text = tools.extract_text(response.content).strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())

        # Remove old memory files (keep MEMORY.md)
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()

        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)

        print(
            f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m"
        )
    except Exception:
        pass


def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    """根据最近用户消息，从记忆库中筛选与当前对话相关的记忆文件名列表。

    双层策略（LLM 优先，关键词回退）：

    LLM 路径（主路径）：
    1. 提取最近 3 条用户消息，拼接为不超过 2000 字符的文本
    2. 列出所有记忆的编号、名称和描述作为 catalog
    3. 让 LLM 返回相关记忆的索引数组（如 [0, 3]）
    4. 将索引映射为文件名，截断到 max_items 条

    关键词回退路径（LLM 调用失败或返回无效数据时）：
    1. 从最近用户消息中提取长度 > 3 的单词做关键词
    2. 在每条记忆的 name + description 中做子串匹配
    3. 取前 max_items 条

    这种双层设计确保即使 LLM 不可用，记忆检索仍然有基本功能。

    Args:
        messages: 完整的对话历史列表。
        max_items: 最多返回的记忆文件数，默认 5 条。

    Returns:
        list[str]: 相关记忆的文件名列表（如 ["user-pref-tabs.md", "project-stack.md"]），
        无相关记忆时返回 []。
    """
    files = list_memory_files()
    if not files:
        return []

    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(getattr(b, "text", ""))
                    for b in content
                    if getattr(b, "type", None) == "text"
                )
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    recent = " ".join(reversed(recent_texts))[:2000]

    if not recent.strip():
        return []

    catalog_lines = []
    for i, f in enumerate(files):
        catalog_lines.append(f"{i}: {f['name']} — {f['description']}")
    catalog = "\n".join(catalog_lines)

    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = tools.extract_text(response.content).strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            selected = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(files):
                    selected.append(files[idx]["filename"])
                    if len(selected) >= max_items:
                        break
            return selected
    except Exception:
        pass

    keywords = [w.lower() for w in recent.split() if len(w) > 3]
    selected = []
    for f in files:
        text = (f["name"] + " " + f["description"]).lower()
        if any(kw in text for kw in keywords):
            selected.append(f["filename"])
            if len(selected) >= max_items:
                break
    return selected


def read_memory_file(filename: str) -> str | None:
    """按文件名读取单条记忆的完整内容（含 frontmatter + 正文）。

    Args:
        filename: 记忆文件名（如 "user-pref-tabs.md"）。

    Returns:
        文件的完整文本；若文件不存在则返回 None。
    """
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()


def load_memories(messages: list) -> str:
    """检索相关记忆并拼接为可注入上下文的 XML 片段。

    这是记忆系统对外的主入口——app.agent_loop 在每轮开始时调用，
    将返回的内容注入到当前用户消息之前。

    工作流程：
    1. 调用 select_relevant_memories 筛选相关记忆文件名
    2. 逐条读取完整内容
    3. 用 <relevant_memories>...</relevant_memories> 包裹拼接

    注入位置（见 app.agent_loop）：在当前轮次首条 user 消息的 content
    之前插入记忆内容，使模型阅读用户输入时同步获得相关上下文。

    Args:
        messages: 完整的对话历史列表，用于 select_relevant_memories 的语义匹配。

    Returns:
        拼接好的 XML 片段字符串；无相关记忆时返回空字符串 ""。
    """
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""

    parts = ["<relevant_memories>"]
    for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


def read_memory_index() -> str:
    """读取 MEMORY.md 索引文件的全部内容。

    供 context.update_context() 调用，间接注入 system prompt 的
    记忆段落。索引按文件名排序，每行一条链接式条目。
    本函数是记忆索引数据的唯一读取入口，外部不应直接操作 MEMORY_INDEX 文件。

    Returns:
        MEMORY.md 的文本内容；文件不存在时返回空字符串 ""。
    """
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""
