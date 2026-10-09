"""Agent Skills（SKILL.md 开放标准）的发现与加载。

为什么要有这一层
----------------
开源生态已经收敛到同一个格式（agentskills.io 规范，Claude Code / Codex CLI /
Gemini CLI / Cursor 通用）：每个技能是一个目录，入口 ``SKILL.md`` 用 YAML
frontmatter 声明 ``name`` / ``description``，正文是完整工作流，可带 scripts/。
把发现器做成标准格式，GitHub 上任何技能丢进目录就能用，不必逐家搬库。

渐进式披露（省 token 的关键）
----------------------------
system prompt 里只放「名称 + 一句话描述」的索引（见 :func:`index_section`）；
模型判断某个技能跟当前任务相关，再用 ``load_skill`` 工具取完整正文。
一个 50 技能的库，常驻上下文只有索引那一点。

三层来源（后者覆盖前者，同名技能以更靠近用户的为准）
----------------------------------------------------
1. builtin：随包捆绑（``agentd/kernel/skills/<name>/``）—— 精选过的开源技能，
   许可见各目录内文件与 skills/README.md；
2. user：``~/.agentd/skills/`` —— 用户自己装的；
3. project：``<cwd>/.agentd/skills/`` —— 项目级，跟着仓库走。

实现刻意不引 pyyaml：frontmatter 实际只用得上 ``key: value`` 和
``key: |``（块标量）两种形态，几十行解析器换零依赖，划算。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# 索引里每条描述的截断长度：description 写几百词的技能（官方仓库就有）会
# 把索引撑爆，一句话足够路由 —— 全文反正要用 load_skill 拿。
_DESC_MAX = 160
# 索引整体预算（字符）：技能再多也不能把 system prompt 吃穿。
_INDEX_BUDGET_CHARS = 2400


@dataclass(frozen=True)
class Skill:
    """一个已发现的技能。``path`` 指向技能目录（SKILL.md 的父目录）。"""

    name: str
    description: str
    path: Path
    source: str  # builtin / user / project


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """把 SKILL.md 拆成 (frontmatter 字典, 正文)。

    只支持规范的常用子集：
        ---
        name: some-skill
        description: 一句话
        description: |      ← 块标量（多行）
          第一行
        ---
    没有 frontmatter / 解析失败返回 ({}, 原文) —— 技能格式坏了不该让发现器崩，
    跳过它就行（调用方据 name 为空丢弃）。
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    head = text[3:end].strip("\n")
    body = text[end + 4:].lstrip("\n")
    fields: dict[str, str] = {}
    key = ""
    block_mode = False
    block_lines: list[str] = []
    for raw in head.splitlines():
        if block_mode:
            if raw[:1] in (" ", "\t"):
                block_lines.append(raw.strip())
                continue
            fields[key] = " ".join(x for x in block_lines if x)
            block_mode = False
            block_lines = []
        m = re.match(r"^([A-Za-z][\w-]*)\s*:\s*(.*)$", raw)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2).strip()
        if value in ("|", ">", "|-", ">-"):
            block_mode = True
            block_lines = []
            continue
        fields[key] = value.strip().strip("'\"")
    if block_mode:
        fields[key] = " ".join(x for x in block_lines if x)
    return fields, body


def _skill_dirs(cwd: str | Path | None) -> list[tuple[Path, str]]:
    """三层来源，按优先级从低到高排列。不存在/建不了的目录跳过。"""
    dirs: list[tuple[Path, str]] = [
        (Path(__file__).resolve().parent / "skills", "builtin"),
        (Path.home() / ".agentd" / "skills", "user"),
    ]
    if cwd:
        try:
            dirs.append((Path(cwd).expanduser().resolve() / ".agentd" / "skills", "project"))
        except OSError:  # pragma: no cover - cwd 怪异时只丢项目层
            pass
    return dirs


def discover(cwd: str | Path | None = None) -> dict[str, Skill]:
    """扫全部来源，返回 {name: Skill}。坏技能静默跳过（发现器不能挂）。"""
    found: dict[str, Skill] = {}
    for root, source in _skill_dirs(cwd):
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for entry in entries:
            skill_md = root / entry / "SKILL.md"
            try:
                if not skill_md.is_file():
                    continue
                fields, _ = parse_frontmatter(
                    skill_md.read_text(encoding="utf-8", errors="replace")
                )
                name = str(fields.get("name") or "").strip()
                description = str(fields.get("description") or "").strip()
            except OSError:
                continue
            if not name:
                continue
            found[name] = Skill(
                name=name, description=description, path=root / entry, source=source
            )
    return found


def index_section(cwd: str | Path | None = None) -> str:
    """system prompt 里的「可用技能」段。没有技能时返回空串（compose_system 跳过）。

    预算护栏：每条描述截到 _DESC_MAX，整体超过 _INDEX_BUDGET_CHARS 就停止追加
    并留一行说明 —— 索引是常驻开销，宁可让模型用 load_skill 深挖。
    """
    skills = discover(cwd)
    if not skills:
        return ""
    lines: list[str] = [
        "【可用技能】以下技能可用 load_skill 工具加载完整工作流；"
        "任务与描述相关时先加载再动手："
    ]
    used = len(lines[0])
    truncated = False
    for name, skill in sorted(skills.items()):
        desc = skill.description.strip().replace("\n", " ")
        if len(desc) > _DESC_MAX:
            desc = desc[: _DESC_MAX - 1] + "…"
        line = f"- {name}（{skill.source}）：{desc}"
        if used + len(line) > _INDEX_BUDGET_CHARS:
            truncated = True
            break
        lines.append(line)
        used += len(line)
    if truncated:
        lines.append(f"…（其余技能未列入索引，可用 load_skill 按名加载；已知名称见技能目录）")
    return "\n".join(lines)


def load_body(name: str, cwd: str | Path | None = None) -> str:
    """取一个技能的完整 SKILL.md 正文（load_skill 工具的落点）。

    找不到时返回带候选名单的错误文本 —— 模型看到名单能自我纠正，不用再来回猜。
    正文过大时截断（与工具输出上限同一量级的二次保险）。
    """
    skills = discover(cwd)
    skill = skills.get(name)
    if skill is None:
        available = ", ".join(sorted(skills)) or "（当前没有已安装的技能）"
        return f"错误：没有叫 {name!r} 的技能。可用的有：{available}"
    try:
        text = (skill.path / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"错误：技能 {name} 的 SKILL.md 读不出来：{exc}"
    # 护栏：个别技能正文带大附录，直接灌会挤占上下文。10KB 足够任何工作流正文。
    if len(text) > 10_000:
        text = text[:10_000] + "\n…（正文过长已截断；完整文件：" + str(skill.path / "SKILL.md") + "）"
    extra = f"（技能目录：{skill.path}；如含 scripts/ 子目录，可用 run_command 执行其中的脚本）"
    return f"{text}\n\n{extra}"
