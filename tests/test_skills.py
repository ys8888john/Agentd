"""Agent Skills（SKILL.md 标准）发现与加载的测试。

三层来源（builtin < user < project，同名覆盖）+ 索引 + load_body。
user 层依赖 Path.home()，测试里不碰它：用 builtin（真实捆绑技能）与
project（tmp_path）两层就能覆盖发现、覆盖、索引、加载四条链路。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentd.kernel import skills
from agentd.kernel.skills import Skill, discover, index_section, load_body, parse_frontmatter

# 捆绑技能清单 —— 这份名单本身就是契约：删技能/改 名 要有意识地更新这里
BUILTIN = {
    "systematic-debugging",
    "verification-before-completion",
    "test-driven-development",
    "writing-plans",
    "skill-creator",
}


# ---------------------------------------------------------------------------
# frontmatter 解析
# ---------------------------------------------------------------------------


def test_parse_frontmatter_simple():
    fields, body = parse_frontmatter(
        "---\nname: foo\ndescription: 做某件事\n---\n\n# 正文\n内容"
    )
    assert fields == {"name": "foo", "description": "做某件事"}
    assert body.startswith("# 正文")


def test_parse_frontmatter_block_scalar():
    """description: | 的多行块标量要拼成一行 —— 索引只要一句话。"""
    text = "---\nname: foo\ndescription: |\n  第一行说明\n  第二行说明\n---\n正文"
    fields, _ = parse_frontmatter(text)
    assert fields["description"] == "第一行说明 第二行说明"


def test_parse_frontmatter_missing_returns_original():
    fields, body = parse_frontmatter("# 没有 frontmatter\n正文")
    assert fields == {}
    assert body.startswith("# 没有 frontmatter")


def test_parse_frontmatter_quoted_value():
    fields, _ = parse_frontmatter('---\nname: "foo-bar"\ndescription: \'x: y\'\n---\n')
    assert fields["name"] == "foo-bar"
    assert fields["description"] == "x: y"


# ---------------------------------------------------------------------------
# 发现与覆盖
# ---------------------------------------------------------------------------


def test_builtin_skills_are_discovered():
    """捆绑技能是包的一部分，发现器必须能找到 —— 少一个就是打包配置坏了。"""
    found = discover()
    missing = BUILTIN - set(found)
    assert not missing, f"捆绑技能没被发现：{missing}（检查 agentd/kernel/skills/ 是否随包发布）"
    assert all(found[name].source == "builtin" for name in BUILTIN)


def test_project_layer_overrides_builtin(tmp_path: Path):
    """同名技能：项目层覆盖内置层 —— 用户改工作流不必动包。"""
    proj = tmp_path / ".agentd" / "skills" / "writing-plans"
    proj.mkdir(parents=True)
    (proj / "SKILL.md").write_text(
        "---\nname: writing-plans\ndescription: 项目定制版计划写法\n---\n项目正文",
        encoding="utf-8",
    )
    found = discover(tmp_path)
    assert found["writing-plans"].source == "project"
    assert found["writing-plans"].description == "项目定制版计划写法"


def test_broken_skill_is_skipped_silently(tmp_path: Path):
    """坏掉的技能（没有 name / 没有 SKILL.md）不能让发现器崩，跳过即可。"""
    bad = tmp_path / ".agentd" / "skills" / "broken"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text("没有 frontmatter", encoding="utf-8")
    (tmp_path / ".agentd" / "skills" / "empty").mkdir(parents=True)
    found = discover(tmp_path)
    assert "broken" not in found and "empty" not in found
    assert BUILTIN <= set(found)  # 其余照常


# ---------------------------------------------------------------------------
# 索引与加载
# ---------------------------------------------------------------------------


def test_index_contains_names_and_description_snippet():
    idx = index_section()
    assert "load_skill" in idx
    for name in BUILTIN:
        assert name in idx


def test_index_respects_description_truncation(tmp_path: Path):
    """超长 description 在索引里截断 —— 路由只要一句话，全文走 load_skill。"""
    d = tmp_path / ".agentd" / "skills" / "long-desc"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: long-desc\ndescription: " + "详" * 500 + "\n---\n正文",
        encoding="utf-8",
    )
    idx = index_section(tmp_path)
    line = [ln for ln in idx.splitlines() if ln.startswith("- long-desc")][0]
    assert "详" * 500 not in line
    assert len(line) < 400


def test_load_body_returns_full_skill():
    body = load_body("systematic-debugging")
    assert "# Systematic Debugging" in body or "Systematic" in body
    assert "技能目录" in body  # 附带目录信息（scripts 提示）


def test_load_body_missing_lists_candidates():
    out = load_body("不存在的技能")
    assert "错误" in out
    for name in BUILTIN:
        assert name in out  # 候选名单让模型能自我纠正


def test_load_body_project_override_wins(tmp_path: Path):
    proj = tmp_path / ".agentd" / "skills" / "writing-plans"
    proj.mkdir(parents=True)
    (proj / "SKILL.md").write_text(
        "---\nname: writing-plans\ndescription: 定制\n---\n项目版正文",
        encoding="utf-8",
    )
    assert "项目版正文" in load_body("writing-plans", tmp_path)


# ---------------------------------------------------------------------------
# load_skill 工具（走 NativeToolbox 的完整调用面）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_skill_tool_lists_and_loads(tmp_path: Path):
    from agentd.kernel.tools import NativeToolbox

    box = NativeToolbox(cwd=tmp_path)
    listing = await box.call("load_skill", "{}")
    assert "可用技能" in listing and "load_skill" in listing

    loaded = await box.call("load_skill", '{"name": "verification-before-completion"}')
    assert "Verification" in loaded or "verification" in loaded.lower()


@pytest.mark.asyncio
async def test_load_skill_tool_unknown_name_gives_candidates(tmp_path: Path):
    from agentd.kernel.tools import NativeToolbox

    box = NativeToolbox(cwd=tmp_path)
    out = await box.call("load_skill", '{"name": "nope"}')
    assert "错误" in out and "writing-plans" in out


# ---------------------------------------------------------------------------
# 打包安全：内置技能目录里不允许混入专有内容
# ---------------------------------------------------------------------------


def test_no_source_available_document_skills_vendored():
    """anthropics/skills 的 docx/pdf/pptx/xlsx 是 source-available，绝不能进包。"""
    bundled_root = Path(skills.__file__).parent / "skills"
    for forbidden in ("docx", "pdf", "pptx", "xlsx"):
        assert not (bundled_root / forbidden).exists(), (
            f"{forbidden} 是 source-available 技能，不许捆绑进本仓库"
        )


def test_skill_dataclass_frozen():
    s = Skill(name="x", description="y", path=Path("."), source="builtin")
    with pytest.raises(Exception):
        s.name = "z"  # type: ignore[misc]
