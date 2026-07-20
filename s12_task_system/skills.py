"""
Skill 管理模块 —— 扫描、注册和检索可用的 Agent 技能。

每个技能以子目录形式存放在 SKILLS_DIR 下：
    skills/
      my-skill/
        SKILL.md   ← YAML frontmatter（name, description）+ Markdown 正文

模块导入时调用 _scan_skills() 一次性遍历文件系统填充 SKILL_REGISTRY，
后续查询命中内存缓存，不需要再访问磁盘。

函数接口：
  list_skills()            → str            所有技能的 Markdown 列表
  get_skill(name)          → dict | None    按名 O(1) 查找（name / description / content）
数据接口：
  SKILL_REGISTRY           → dict[str, dict] tools.load_skill 直接从此字典取值
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
    """
    if not text.startswith("---"):
        return {}, text
    # split("---", 2)：只分割前两组 "---"，避免正文中的 "---" 干扰
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
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

    如果 SKILL.md 不存在则跳过该目录。
    """
    if not SKILLS_DIR.exists():
        return
    # sorted 保证扫描顺序稳定，避免不同平台/文件系统的遍历顺序差异
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            # name 优先取 frontmatter 中的 name 字段，其次用目录名
            name = meta.get("name", d.name)
            # description 优先取 frontmatter，其次取正文首行（去掉 # 标记）
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}


# 模块导入时立即扫描，确保 SKILL_REGISTRY 在首次使用前已就绪
_scan_skills()


def list_skills() -> str:
    """以 Markdown 列表形式返回所有已注册技能的名称和描述。"""
    return "\n".join(
        f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values()
    )


def get_skill(name: str) -> dict | None:
    """按名称从全局注册表中查找技能，返回包含 name / description / content 的字典。

    O(1) 字典查找，无需访问文件系统。未找到时返回 None，
    调用方（tools.load_skill）负责将 None 转为错误提示字符串返回给模型。
    """
    return SKILL_REGISTRY.get(name)
