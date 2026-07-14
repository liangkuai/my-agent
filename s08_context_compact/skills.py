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
            SKILL.md   ← 必须，包含 YAML frontmatter + Markdown 正文

    注册逻辑：
    - name 优先取 frontmatter 中的 name 字段，缺失时用目录名兜底。
    - description 优先取 frontmatter 中的 description 字段，缺失时用正文首行
      （去掉开头的 # 标记）兜底。
    - content 保存 SKILL.md 的完整原文，供 load_skill 按名返回。

    如果 SKILL.md 不存在则跳过该目录。结果写入全局 SKILL_REGISTRY，
    模块导入时自动执行一次，后续查询无需再访问文件系统。
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
    """以 Markdown 列表形式返回所有已注册技能的名称和描述。

    供 app.build_system() 拼入 system prompt，让模型了解当前可加载哪些技能。
    注意：当前 system prompt 在模块加载时一次性构建（见 app.py 的 SYSTEM），
    因此本函数只在启动时调用一次；如果后续改为动态构建 system prompt，
    则需要每次重新调用。
    """
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())


def get_skill(name: str) -> dict | None:
    """按名称从全局注册表中查找技能，返回包含 name / description / content 的字典。

    O(1) 字典查找，无需访问文件系统。未找到时返回 None，
    调用方（tools.load_skill）负责将 None 转为错误提示字符串返回给模型。
    """
    return SKILL_REGISTRY.get(name)
