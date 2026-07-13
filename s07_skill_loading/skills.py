"""
技能系统 —— 扫描 skills/ 目录并将每个 SKILL.md 注册到内存注册表。

设计要点：
1. 模块导入时一次性扫描，后续查询走内存（零文件系统开销）。
2. 每个技能是 skills/<name>/SKILL.md，含 YAML frontmatter + Markdown 正文。
3. load_skill：按名返回完整原文，供模型按需加载。
4. list_skills：只返回名称和描述，节省系统提示 token。
"""

import yaml

from constant import SKILLS_DIR

# 全局技能注册表：模块加载时由 _scan_skills() 一次性填充，
# 后续查询直接走内存，避免每次 list 都扫一次文件系统。
SKILL_REGISTRY: dict[str, dict] = {}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 SKILL.md 中的 YAML frontmatter。

    约定格式：
        ---
        name: my-skill
        description: 一句话描述
        ---
        正文内容...

    返回 (meta_dict, body_text)。如果文件不以 "---" 开头或解析失败，
    返回 ({}, 原文)，调用方退化使用文件名和首行标题作为名称和描述。

    仅分割前两组 "---" 避免正文中的 YAML 分隔符干扰；
    使用 yaml.safe_load 而非 yaml.load 以禁用任意 Python 对象反序列化。
    """
    if not text.startswith("---"):
        return {}, text
    # split("---", 2)：只分割前两组 "---"，避免正文中的 "---" 干扰
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        # safe_load 只解析基础 YAML 类型（dict/list/str/int/float/bool/null），
        # 禁用 !!python/object 等危险标签，防止技能文件反序列化攻击。
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


def _scan_skills() -> None:
    """扫描 SKILLS_DIR 下的所有子目录，加载每个目录中的 SKILL.md 并注册。

    每个技能的目录结构：
        skills/
          my-skill/
            SKILL.md   ← 必须，包含 frontmatter + 正文

    如果 SKILL.md 不存在则静默跳过该目录（允许放 README 或其他辅助文件）。

    注册表条目结构：
        {"name": str, "description": str, "content": str}
    """
    if not SKILLS_DIR.exists():
        return
    # sorted 保证扫描顺序稳定，避免不同平台/文件系统的遍历顺序差异
    # 导致 list_skills() 输出因平台而异。
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            # name 优先级：frontmatter 的 name 字段 > 目录名
            name = meta.get("name", d.name)
            # description 优先级：frontmatter 的 description > 正文首行（去掉 #）
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}


# 模块导入时立即扫描，确保 SKILL_REGISTRY 在首次使用前已就绪
_scan_skills()


def list_skills() -> str:
    """以 Markdown 列表形式返回所有已注册技能的名称和描述。

    用于组装系统提示（见 app.build_system），让模型知道有哪些可用技能。
    不含正文内容——模型需要时通过 load_skill 工具按需加载。
    """
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())
