"""原生工具层（kernel/tools.py）测试。

分六块：
1. 七个本地工具各自的行为（真文件系统，全在 tmp_path 里）；
2. 路径边界与 profile（能不能越出 cwd、read_only 能不能挡住写）；
3. needs_approval 的判定矩阵 + AgentMode 的审批/kind 端到端；
4. ACP 映射：_ACP_KIND 必须覆盖 kernel 产出的每一个 ToolKind（漏一个 = 客户端静默卡死）；
5. 内核接线；6. 联网工具的解析 / 编码 / 错误路径（真发 HTTP 的部分在
tests/test_web_tools_live.py，需要显式开 AGENTD_LIVE_WEB=1）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from agentd.contracts import ToolCallDone, ToolCallStart, ToolCallStart as _TCS
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.modes.agent import AgentMode
from agentd.kernel.modes.base import ModeContext
from agentd.kernel.models import Message
from agentd.kernel.tools import (
    ALL_TOOL_NAMES,
    ApprovalRequest,
    NativeToolbox,
    TOOL_PROFILES,
    _decode,
    _html_to_text,
    _is_tls_error,
    _looks_textual,
    _parse_bing,
    _unwrap_bing_url,
    needs_approval,
    workspace_brief,
)
from agentd.transports.acp_stdio import _ACP_KIND

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def box(tmp_path: Path, **kw) -> NativeToolbox:
    """默认把 root 钉在 tmp_path 上 —— 测试绝不允许碰到仓库里的真文件。"""
    kw.setdefault("cwd", tmp_path)
    return NativeToolbox(**kw)


# ---------------------------------------------------------------------------
# 1) 各本地工具
# ---------------------------------------------------------------------------


# ---- list_dir ----


async def test_list_dir_shows_dirs_before_files(tmp_path):
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x", encoding="utf-8")
    (tmp_path / "empty-dir").mkdir()   # 空目录：glob 看不见它，list_dir 必须看得见
    tb = box(tmp_path)

    out = await tb.call("list_dir", "{}")
    lines = out.splitlines()
    # 目录排在文件前面且带 / 后缀，模型一眼能分清"能往里钻"和"能读内容"
    entries = [line.strip() for line in lines[1:]]
    assert entries == ["empty-dir/", "src/", "notes.md"]
    # 空目录不会被 _walk_tree 漏掉 —— 这正是补这个工具的主要原因
    assert "empty-dir/" in out

    # 递归时目录名不许重复：父亲已经列过 src/，孩子只能多一层缩进
    nested = await tb.call("list_dir", json.dumps({"depth": 2}))
    assert nested.count("src/") == 1
    assert "  main.py" in nested


async def test_list_dir_depth_controls_recursion(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.txt").write_text("x", encoding="utf-8")
    tb = box(tmp_path)

    shallow = await tb.call("list_dir", "{}")
    assert "a/" in shallow and "b/" not in shallow

    deeper = await tb.call("list_dir", json.dumps({"depth": 3}))
    assert "b/" in deeper


async def test_list_dir_hides_noise_and_hidden_by_default(tmp_path):
    (tmp_path / "keep.py").write_text("x", encoding="utf-8")
    (tmp_path / ".env").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    tb = box(tmp_path)

    out = await tb.call("list_dir", "{}")
    assert "keep.py" in out
    assert "node_modules" not in out
    assert ".env" not in out

    # 显式要求时它们必须能出来 —— "默认过滤"不等于"根本拿不到"
    everything = await tb.call(
        "list_dir", json.dumps({"include_hidden": True, "include_ignored": True})
    )
    assert "node_modules/" in everything
    assert ".env" in everything


async def test_workspace_brief_is_short_and_actionable(tmp_path):
    """system prompt 里的那份目录清单：必须短、必须带 cwd、必须过滤噪音。"""
    (tmp_path / "README.md").write_text("x", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x", encoding="utf-8")

    brief = workspace_brief(tmp_path)
    assert str(tmp_path) in brief           # 绝对路径：模型要靠它构造其它路径
    # depth=2 ⇒ 子目录里的文件也要露个头（"src/ 下有 main.py"是最有用的一条）
    assert "src/" in brief and "main.py" in brief
    assert "node_modules" not in brief      # 噪音目录不进 system prompt
    assert "1 个子目录" in brief            # 计数得跟画出来的树对得上
    # 每一轮的固定开销，不能失控
    assert len(brief.splitlines()) <= 48


async def test_workspace_brief_ok_when_cwd_missing(tmp_path):
    # 目录被删 / 没权限：环境说明掉了可以，抛异常把整轮搞挂不行
    assert workspace_brief(tmp_path / "nope") == ""


async def test_list_dir_errors(tmp_path):
    tb = box(tmp_path)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")

    # 拿文件当目录用：报错还要把正确的工具指出来（模型下一步才知道该干嘛）
    out = await tb.call("list_dir", json.dumps({"path": "a.txt"}))
    assert out.startswith("[错误]") and "read_file" in out

    out = await tb.call("list_dir", json.dumps({"path": "nope"}))
    assert out.startswith("[错误]") and "不存在" in out

    # 越界同其它工具一样必须拦住
    out = await tb.call("list_dir", json.dumps({"path": "../"}))
    assert out.startswith("[错误]") and "越界" in out


async def test_list_dir_reports_capacity_and_caps(tmp_path):
    for i in range(20):
        (tmp_path / f"f{i:02}.txt").write_text("x", encoding="utf-8")
    tb = box(tmp_path)

    out = await tb.call("list_dir", json.dumps({"max_results": 5}))
    assert "已达 5 条上限" in out
    head = out.splitlines()[0]
    assert "20 个文件" in head  # 总数照实报，别让模型以为目录里只有 5 个


async def test_read_file_numbers_lines_and_respects_offset(tmp_path):
    (tmp_path / "a.txt").write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
    tb = box(tmp_path)

    out = await tb.call("read_file", json.dumps({"path": "a.txt"}))
    assert "共 4 行" in out
    assert "     1\tl1" in out
    assert "     4\tl4" in out

    out = await tb.call("read_file", json.dumps({"path": "a.txt", "offset": 3, "limit": 1}))
    assert "第 3-3 行" in out
    assert "l3" in out
    assert "l4" not in out


async def test_read_file_errors_are_prefixed(tmp_path):
    tb = box(tmp_path)
    assert (await tb.call("read_file", json.dumps({"path": "nope.txt"}))).startswith("[错误]")
    (tmp_path / "d").mkdir()
    assert (await tb.call("read_file", json.dumps({"path": "d"}))).startswith("[错误]")
    assert (await tb.call("read_file", "{}")).startswith("[错误]")


async def test_glob_lists_relative_paths_newest_first(tmp_path):
    (tmp_path / "src").mkdir()
    old = tmp_path / "src" / "old.py"
    old.write_text("", encoding="utf-8")
    new = tmp_path / "src" / "new.py"
    new.write_text("", encoding="utf-8")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    out = await box(tmp_path).call("glob", json.dumps({"pattern": "**/*.py"}))
    assert "匹配 2 个文件" in out
    lines = [ln for ln in out.splitlines() if ln.endswith(".py")]
    assert lines == ["src/new.py", "src/old.py"]  # 新的在前，且是相对路径


async def test_glob_rejects_absolute_pattern(tmp_path):
    """Path.glob 不吃绝对 pattern，报的是 NotImplementedError —— 要接住并说人话。"""
    out = await box(tmp_path).call("glob", json.dumps({"pattern": str(tmp_path / "*")}))
    assert out.startswith("[错误]")
    assert "绝对路径" in out


async def test_glob_no_match_is_not_an_error(tmp_path):
    out = await box(tmp_path).call("glob", json.dumps({"pattern": "*.rs"}))
    assert not out.startswith("[错误]")
    assert "没有匹配" in out


async def test_grep_returns_file_line_and_content(tmp_path):
    (tmp_path / "m.py").write_text("import os\n\ndef handle_x():\n    pass\n", encoding="utf-8")
    out = await box(tmp_path).call("grep", json.dumps({"pattern": r"def handle_\w+"}))
    assert "m.py:3: def handle_x():" in out


async def test_grep_skips_noise_dirs_and_honors_glob(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hit.py").write_text("target\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "hit.py").write_text("target\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("target\n", encoding="utf-8")
    (tmp_path / "note.md").write_text("target\n", encoding="utf-8")

    tb = box(tmp_path)
    out = await tb.call("grep", json.dumps({"pattern": "target"}))
    assert "keep.py" in out
    assert "note.md" in out
    assert ".git" not in out and "node_modules" not in out

    only_py = await tb.call("grep", json.dumps({"pattern": "target", "glob": "*.py"}))
    assert "keep.py" in only_py
    assert "note.md" not in only_py


async def test_grep_bad_regex_and_missing_pattern(tmp_path):
    tb = box(tmp_path)
    assert "正则表达式非法" in await tb.call("grep", json.dumps({"pattern": "([unclosed"}))
    assert (await tb.call("grep", "{}")).startswith("[错误]")


async def test_grep_ignore_case(tmp_path):
    (tmp_path / "f.txt").write_text("Hello\n", encoding="utf-8")
    tb = box(tmp_path)
    assert "没有匹配" in await tb.call("grep", json.dumps({"pattern": "hello"}))
    assert "f.txt" in await tb.call("grep", json.dumps({"pattern": "hello", "ignore_case": True}))


async def test_write_file_creates_parents_and_reports_action(tmp_path):
    tb = box(tmp_path)
    out = await tb.call("write_file", json.dumps({"path": "deep/nest/x.txt", "content": "a\nb"}))
    assert "新建" in out and "2 行" in out
    assert (tmp_path / "deep" / "nest" / "x.txt").read_text(encoding="utf-8") == "a\nb"

    out = await tb.call("write_file", json.dumps({"path": "deep/nest/x.txt", "content": "z"}))
    assert "覆盖" in out


async def test_write_file_requires_content_explicitly(tmp_path):
    """空字符串是合法内容，缺字段不是 —— 两者要区分开，否则会静默清空文件。"""
    tb = box(tmp_path)
    assert (await tb.call("write_file", json.dumps({"path": "x.txt"}))).startswith("[错误]")
    out = await tb.call("write_file", json.dumps({"path": "x.txt", "content": ""}))
    assert "新建" in out
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == ""


async def test_edit_replaces_unique_match(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("def f():\n    return 1\n", encoding="utf-8")
    out = await box(tmp_path).call(
        "edit", json.dumps({"path": "c.py", "old_string": "return 1", "new_string": "return 2"})
    )
    assert "替换 1 处" in out
    assert f.read_text(encoding="utf-8") == "def f():\n    return 2\n"


async def test_edit_refuses_ambiguous_match_then_replace_all(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("x = 1\ny = 1\n", encoding="utf-8")
    tb = box(tmp_path)

    out = await tb.call("edit", json.dumps({"path": "c.py", "old_string": "= 1", "new_string": "= 2"}))
    assert out.startswith("[错误]") and "2 处" in out
    assert f.read_text(encoding="utf-8") == "x = 1\ny = 1\n"  # 被拒时绝不能动文件

    out = await tb.call(
        "edit",
        json.dumps({"path": "c.py", "old_string": "= 1", "new_string": "= 2", "replace_all": True}),
    )
    assert "替换 2 处" in out
    assert f.read_text(encoding="utf-8") == "x = 2\ny = 2\n"


async def test_edit_reports_not_found_and_missing_file(tmp_path):
    (tmp_path / "c.py").write_text("abc\n", encoding="utf-8")
    tb = box(tmp_path)
    assert "找不到 old_string" in await tb.call(
        "edit", json.dumps({"path": "c.py", "old_string": "zzz", "new_string": "1"})
    )
    assert (await tb.call(
        "edit", json.dumps({"path": "no.py", "old_string": "a", "new_string": "b"})
    )).startswith("[错误]")
    # 空 old_string 会把每个字符之间都插一遍，必须拒
    assert (await tb.call(
        "edit", json.dumps({"path": "c.py", "old_string": "", "new_string": "b"})
    )).startswith("[错误]")


async def test_run_command_captures_stdout_and_stderr(tmp_path):
    script = tmp_path / "emit.py"
    script.write_text(
        "import sys\nprint('OUT')\nprint('ERR', file=sys.stderr)\n", encoding="utf-8"
    )
    tb = box(tmp_path)
    out = await tb.call(
        "run_command",
        json.dumps({"command": f'"{sys.executable}" "{script}"'}),
    )
    assert out.startswith("exit=0")
    assert "OUT" in out and "[stderr]" in out and "ERR" in out


async def test_run_command_nonzero_exit_is_reported_as_failure(tmp_path):
    tb = box(tmp_path)
    out = await tb.call(
        "run_command", json.dumps({"command": f'"{sys.executable}" -c "raise SystemExit(3)"'})
    )
    assert out.startswith("[错误]")
    assert "退出码 3" in out


async def test_run_command_timeout_kills_and_reports(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    tb = box(tmp_path, timeout=0.6)
    out = await tb.call("run_command", json.dumps({"command": f'"{sys.executable}" "{script}"'}))
    assert out.startswith("[错误]") and "超时" in out


async def test_run_command_uses_cwd_relative_to_session_root(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "marker.txt").write_text("m", encoding="utf-8")
    script = tmp_path / "ls.py"
    script.write_text("import os\nprint(sorted(os.listdir('.')))\n", encoding="utf-8")
    tb = box(tmp_path)
    out = await tb.call(
        "run_command", json.dumps({"command": f'"{sys.executable}" "{script}"', "cwd": "sub"})
    )
    assert "marker.txt" in out


# ---- make_xlsx ----


def test_make_xlsx_produces_a_real_zip_with_the_ooxml_parts():
    """xlsx 本质是 zip：五个必需部件一个都不能少，否则 Excel/openpyxl 打不开。"""
    import io
    import zipfile

    from agentd.kernel.tools import build_xlsx_bytes

    data = build_xlsx_bytes("sheet1", [["航班号", "航司"], ["CA1402", "国航"]])
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
    assert {
        "[Content_Types].xml",
        "_rels/.rels",
        "xl/workbook.xml",
        "xl/_rels/workbook.xml.rels",
        "xl/worksheets/sheet1.xml",
    } <= names


async def test_make_xlsx_roundtrips_through_openpyxl(tmp_path):
    """拿 openpyxl 当裁判读回来：表头 / 数字 / 中文 / 工作表名都要对得上。

    openpyxl 是本机 venv 里已装的独立实现 —— 用它交叉验证比"自己解自己写的 zip"
    可信得多（后者等于用同一套假设自证）。
    """
    openpyxl = pytest.importorskip("openpyxl")
    tb = box(tmp_path)
    out = await tb.call(
        "make_xlsx",
        json.dumps(
            {
                "path": "航班列表",
                "sheet_name": "航班",
                "headers": ["航班号", "航司", "价格"],
                "rows": [["CA1402", "国航", 520], ["3U8899", "川航", 410.5]],
            }
        ),
    )
    assert out.startswith("已新建")
    target = tmp_path / "航班列表.xlsx"  # 后缀被自动补上
    assert target.is_file()

    ws = openpyxl.load_workbook(target).active
    assert ws.title == "航班"
    assert [c.value for c in ws[1]] == ["航班号", "航司", "价格"]
    assert ws["A2"].value == "CA1402"
    assert ws["C2"].value == 520 and isinstance(ws["C2"].value, int)
    assert ws["C3"].value == 410.5 and isinstance(ws["C3"].value, float)


async def test_make_xlsx_does_not_turn_text_into_a_formula(tmp_path):
    """以 = 开头的抓取文本必须原样是字符串 —— 否则就是公式注入。"""
    openpyxl = pytest.importorskip("openpyxl")
    tb = box(tmp_path)
    await tb.call("make_xlsx", json.dumps({"path": "f.xlsx", "rows": [["=1+1"]]}))
    ws = openpyxl.load_workbook(tmp_path / "f.xlsx").active
    assert ws["A1"].value == "=1+1"  # 是字符串，不是公式结果 2


async def test_make_xlsx_accepts_object_rows_in_header_order(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    tb = box(tmp_path)
    await tb.call(
        "make_xlsx",
        json.dumps(
            {
                "path": "o.xlsx",
                "headers": ["a", "b"],
                "rows": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            }
        ),
    )
    ws = openpyxl.load_workbook(tmp_path / "o.xlsx").active
    assert [c.value for c in ws[1]] == ["a", "b"]
    assert [c.value for c in ws[2]] == [1, 2]


async def test_make_xlsx_rejects_out_of_root_and_bad_args(tmp_path):
    tb = box(tmp_path)
    escape = await tb.call(
        "make_xlsx", json.dumps({"path": "../escape.xlsx", "rows": [[1]]})
    )
    assert escape.startswith("[错误]") and "越界" in escape

    assert (await tb.call("make_xlsx", json.dumps({"rows": [[1]]}))).startswith("[错误]")
    assert (await tb.call(
        "make_xlsx", json.dumps({"path": "x.xlsx", "rows": []})
    )).startswith("[错误]")
    assert (await tb.call(
        "make_xlsx", json.dumps({"path": "x.xlsx", "rows": "nope"})
    )).startswith("[错误]")
    assert (await tb.call(
        "make_xlsx", json.dumps({"path": "x.xlsx", "headers": "bad", "rows": [[1]]})
    )).startswith("[错误]")


async def test_make_xlsx_requires_approval_like_other_state_changers(tmp_path):
    """写文件类工具必须弹审批（policy=native 默认行为）。"""
    tb = box(tmp_path)
    binding = tb.binding("make_xlsx")
    assert binding is not None
    assert binding.requires_permission is True
    assert binding.kind == "execute"
    assert "make_xlsx" in TOOL_PROFILES["native"]
    assert "make_xlsx" not in TOOL_PROFILES["read_only"]


async def test_make_xlsx_pads_ragged_rows_to_equal_width(tmp_path):
    """参差的行要补齐成等宽 —— 否则 openpyxl 看到的列数会随最长行漂移。"""
    openpyxl = pytest.importorskip("openpyxl")
    tb = box(tmp_path)
    await tb.call(
        "make_xlsx", json.dumps({"path": "r.xlsx", "rows": [[1, 2, 3], [4]]})
    )
    ws = openpyxl.load_workbook(tmp_path / "r.xlsx").active
    assert ws.max_column == 3
    assert ws["C2"].value is None


# ---------------------------------------------------------------------------
# 2) 边界与 profile
# ---------------------------------------------------------------------------


async def test_path_escape_outside_root_is_rejected(tmp_path):
    """`..` 必须被 resolve 展开后再判，否则 ../ 就成了走后门的通道。"""
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")

    tb = NativeToolbox(cwd=root)
    out = await tb.call("read_file", json.dumps({"path": "../secret.txt"}))
    assert out.startswith("[错误]") and "越界" in out
    assert "top secret" not in out

    # 写同样要拦
    out = await tb.call("write_file", json.dumps({"path": "../evil.txt", "content": "x"}))
    assert out.startswith("[错误]")
    assert not (tmp_path / "evil.txt").exists()


async def test_allow_outside_opens_the_gate(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "outer.txt").write_text("ok", encoding="utf-8")
    tb = NativeToolbox(cwd=root, allow_outside=True)
    out = await tb.call("read_file", json.dumps({"path": "../outer.txt"}))
    assert "ok" in out


async def test_profiles_control_which_tools_exist(tmp_path):
    assert NativeToolbox(cwd=tmp_path, profile="native").names == list(ALL_TOOL_NAMES)
    assert NativeToolbox(cwd=tmp_path, profile="read_only").names == [
        "read_file",
        "list_dir",
        "glob",
        "grep",
    ]
    assert NativeToolbox(cwd=tmp_path, profile="off").names == []
    assert not NativeToolbox(cwd=tmp_path, profile="off").has_tools
    assert TOOL_PROFILES["off"] == () and TOOL_PROFILES["none"] == ()


async def test_disabled_tool_is_unknown(tmp_path):
    tb = NativeToolbox(cwd=tmp_path, profile="read_only")
    out = await tb.call("run_command", json.dumps({"command": "echo hi"}))
    assert out.startswith("[错误]") and "未知工具" in out


async def test_bad_arguments_are_reported_not_raised(tmp_path):
    tb = box(tmp_path)
    assert "不是合法 JSON" in await tb.call("read_file", "{oops")
    assert "必须是 JSON 对象" in await tb.call("read_file", "[1,2]")
    assert (await tb.call("nope", "{}")).startswith("[错误]")


async def test_output_is_truncated(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
    tb = box(tmp_path, max_bytes=500)
    out = await tb.call("read_file", json.dumps({"path": "big.txt"}))
    assert "已截断" in out
    assert len(out) < 900


async def test_tool_schema_shape_matches_openai(tmp_path):
    schema = box(tmp_path).tool_schema()
    assert {t["function"]["name"] for t in schema} == set(ALL_TOOL_NAMES)
    for item in schema:
        assert item["type"] == "function"
        fn = item["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


# ---------------------------------------------------------------------------
# 3) 审批判定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy,kind,requires,destructive,expected",
    [
        # 只读动作在任何策略下都不弹 —— 每个 ls 都弹窗，用户三分钟就学会无脑点允许
        ("native", "read", False, False, False),
        ("native", "search", False, False, False),
        ("all", "read", False, False, False),
        # 原生写/执行类
        ("native", "edit", True, False, True),
        ("native", "execute", True, False, True),
        # MCP 工具没声明 requires，只有 destructiveHint 能拦
        ("native", "execute", False, False, False),
        ("native", "execute", False, True, True),
        ("all", "execute", False, False, True),
        ("all", "edit", False, False, True),
        # none 是无人值守档，全放
        ("none", "execute", True, True, False),
        ("none", "edit", True, False, False),
        # 拼错的策略退回默认（native），不能因为拼错就全放行
        ("nonsense", "edit", True, False, True),
    ],
)
def test_needs_approval_matrix(policy, kind, requires, destructive, expected):
    assert (
        needs_approval(requires=requires, kind=kind, destructive=destructive, policy=policy)
        is expected
    )


# ---------------------------------------------------------------------------
# 4) AgentMode：kind 映射 + 审批 + 与 MCP 共存
# ---------------------------------------------------------------------------


class ScriptedLLM(LLM):
    """按轮产出事件；记录每次收到的 tools 便于断言。"""

    def __init__(self, rounds: list[list]) -> None:
        self._rounds = list(rounds)
        self.seen_tools: list[list[dict] | None] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.seen_tools.append(tools)
        round_events = self._rounds.pop(0) if self._rounds else [LLMText("（脚本耗尽）")]
        for event in round_events:
            yield event


def _ctx(tmp_path, llm, **kw) -> ModeContext:
    kw.setdefault("toolbox", NativeToolbox(cwd=tmp_path))
    return ModeContext(
        session_id="s", run_id="r", llm=llm, history=[Message.user("hi")], cwd=str(tmp_path), **kw
    )


async def test_agent_mode_maps_kind_per_tool(tmp_path):
    """kind 不能再写死 execute：读/搜/写/执行各是各的图标。"""
    (tmp_path / "a.txt").write_text("hi\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            [
                LLMToolCall(id="c1", name="read_file", arguments='{"path":"a.txt"}'),
                LLMToolCall(id="c2", name="glob", arguments='{"pattern":"*.txt"}'),
                LLMToolCall(
                    id="c3",
                    name="write_file",
                    arguments='{"path":"b.txt","content":"x"}',
                ),
            ],
            [LLMText("好了")],
        ]
    )
    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]

    kinds = {e.call_id: e.kind for e in events if isinstance(e, ToolCallStart)}
    assert kinds == {"c1": "read", "c2": "search", "c3": "edit"}
    done = {e.call_id: e.status for e in events if isinstance(e, ToolCallDone)}
    assert done == {"c1": "completed", "c2": "completed", "c3": "completed"}


async def test_agent_mode_merges_native_and_mcp_schema(tmp_path):
    """两条工具来源都要进 tools 数组，且原生排在前面。"""
    llm = ScriptedLLM([[LLMText("ok")]])

    class _Hub:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def tool_schema(self):
            return [
                {"type": "function", "function": {"name": "srv__t", "description": "", "parameters": {}}}
            ]

        def binding(self, name):
            return None

        async def call(self, name, args):
            return ""

    import agentd.kernel.modes.agent as agent_mod

    original = agent_mod.McpHub
    agent_mod.McpHub = _Hub
    try:
        [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]
    finally:
        agent_mod.McpHub = original

    names = [t["function"]["name"] for t in llm.seen_tools[0]]
    assert names[: len(ALL_TOOL_NAMES)] == list(ALL_TOOL_NAMES)
    assert names[-1] == "srv__t"


async def test_agent_mode_asks_approval_and_refuses_when_denied(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="write_file", arguments='{"path":"x.txt","content":"boom"}')],
            [LLMText("好")],
        ]
    )
    seen: list[ApprovalRequest] = []

    async def deny(req: ApprovalRequest) -> bool:
        seen.append(req)
        return False

    events = [
        e
        async for e in AgentMode().run(_ctx(tmp_path, llm, approve=deny), "hi")
    ]

    assert [r.tool for r in seen] == ["write_file"]
    assert seen[0].kind == "edit"
    assert "x.txt" in seen[0].detail  # 弹窗摘要来自参数
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "cancelled"
    assert not (tmp_path / "x.txt").exists()  # 被拒就绝不能落盘


async def test_agent_mode_executes_when_approved(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="write_file", arguments='{"path":"x.txt","content":"ok"}')],
            [LLMText("好")],
        ]
    )
    asked: list[str] = []

    async def allow(req: ApprovalRequest) -> bool:
        asked.append(req.tool)
        return True

    [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=allow), "hi")]
    assert asked == ["write_file"]
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "ok"


async def test_agent_mode_does_not_ask_for_read_only_tools(tmp_path):
    (tmp_path / "a.txt").write_text("hi\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments='{"path":"a.txt"}')],
            [LLMText("好")],
        ]
    )

    async def boom(req: ApprovalRequest) -> bool:
        raise AssertionError("只读工具不该触发审批")

    [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=boom), "hi")]


async def test_agent_mode_tool_failure_marks_status_failed(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments='{"path":"missing.txt"}')],
            [LLMText("文件不存在")],
        ]
    )
    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "failed"
    assert done[0].output.startswith("[错误]")


async def test_agent_mode_unknown_tool_still_closes_the_card(tmp_path):
    """模型幻觉出一个工具名时，start/done 必须配对发出去，否则客户端卡片永远转圈。"""
    llm = ScriptedLLM(
        [[LLMToolCall(id="c1", name="totally_made_up", arguments="{}")], [LLMText("好")]]
    )
    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]
    start = [e for e in events if isinstance(e, ToolCallStart)]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert len(start) == len(done) == 1
    assert start[0].kind == "other"
    assert done[0].status == "failed" and "未知工具" in done[0].output


# --- 方案③：非只读 MCP 工具也走审批（不再只信服务端 destructiveHint） ---


class _McpBinding:
    def __init__(self, read_only: bool):
        self.tool = "tool"
        self.read_only = read_only

    async def call(self, name, args):  # pragma: no cover - 被拒/只读分支用不到
        return "ok"


def _mcp_hub_factory(target: str, readonly: bool):
    """返回一个可被 McpHub 替换的类：命中 target 的返回带 read_only 的 binding。"""

    class _Hub:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def binding(self, name):
            return _McpBinding(read_only=readonly) if name == target else None

        def tool_schema(self):
            return []

        async def call(self, name, args):
            return "ok"

    return _Hub


async def test_mcp_execute_tool_requires_approval(tmp_path):
    """非只读 MCP 工具（如 browser 开页）必须弹审批，被拒就不执行。"""
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="browser__open_url", arguments='{"url":"https://example.com"}')],
            [LLMText("好")],
        ]
    )
    seen: list[ApprovalRequest] = []

    async def deny(req: ApprovalRequest) -> bool:
        seen.append(req)
        return False

    import agentd.kernel.modes.agent as agent_mod

    original = agent_mod.McpHub
    agent_mod.McpHub = _mcp_hub_factory("browser__open_url", readonly=False)
    try:
        events = [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=deny), "hi")]
    finally:
        agent_mod.McpHub = original

    assert [r.tool for r in seen] == ["browser__open_url"]
    assert seen[0].kind == "execute"
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "cancelled"


async def test_mcp_readonly_tool_skips_approval(tmp_path):
    """只读 MCP 工具（read_only_hint=True）不弹审批，直接执行。"""
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="time__now", arguments="{}")],
            [LLMText("好")],
        ]
    )

    async def boom(req: ApprovalRequest) -> bool:
        raise AssertionError("只读 MCP 工具不该触发审批")

    import agentd.kernel.modes.agent as agent_mod

    original = agent_mod.McpHub
    agent_mod.McpHub = _mcp_hub_factory("time__now", readonly=True)
    try:
        events = [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=boom), "hi")]
    finally:
        agent_mod.McpHub = original

    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "completed"


async def test_approval_exception_is_treated_as_refusal(tmp_path):
    """审批通道坏了必须往"拒绝"倒 —— 反过来就是安全漏洞。"""
    llm = ScriptedLLM(
        [[LLMToolCall(id="c1", name="write_file", arguments='{"path":"x.txt","content":"a"}')],
         [LLMText("好")]]
    )

    async def broken(req: ApprovalRequest) -> bool:
        raise RuntimeError("客户端炸了")

    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=broken), "hi")]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "cancelled"
    assert not (tmp_path / "x.txt").exists()


# ---------------------------------------------------------------------------
# 5) 内核接线
# ---------------------------------------------------------------------------


async def test_kernel_enables_native_tools_by_default(tmp_path):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm)
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]

    names = [t["function"]["name"] for t in llm.seen_tools[0]]
    assert set(ALL_TOOL_NAMES) <= set(names)


async def test_kernel_can_turn_native_tools_off(tmp_path):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm, native_tools="off")
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    assert llm.seen_tools[0] is None  # 无工具 ⇒ 与 single 等价


async def test_kernel_read_only_profile(tmp_path):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm, native_tools="read_only")
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    names = {t["function"]["name"] for t in llm.seen_tools[0]}
    assert names == {"read_file", "list_dir", "glob", "grep"}


async def test_kernel_tool_runs_against_session_cwd(tmp_path):
    (tmp_path / "only-here.txt").write_text("x", encoding="utf-8")
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="glob", arguments='{"pattern":"*.txt"}')],
            [LLMText("好")],
        ]
    )
    kernel = AgentKernel(llm=llm)
    session_id = await kernel.create_session(cwd=str(tmp_path))

    events = [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert "only-here.txt" in done[0].output

    # 库里一行一行是：user → assistant(举手要调工具) → 工具卡片 Role="tool_record"
    # （UI 回放用）→ role="tool"（工具真实输出，下一轮要接着给模型看）→ assistant 结论。
    # 中间那两行是同一件事的两种存法：一个给 UI，一个给模型。
    history = await kernel.history(session_id)
    assert [m.role for m in history] == ["user", "assistant", "tool_record", "tool", "assistant"]
    tool_msg = history[3]
    assert tool_msg.tool_call_id == "c1"
    assert "only-here.txt" in tool_msg.content


async def test_kernel_approve_callback_is_used(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="run_command", arguments='{"command":"echo hi"}')],
            [LLMText("好")],
        ]
    )
    kernel = AgentKernel(llm=llm)
    session_id = await kernel.create_session(cwd=str(tmp_path))
    asked: list[str] = []

    async def approve(req: ApprovalRequest) -> bool:
        asked.append(req.tool)
        return True

    [e async for e in kernel.handle(session_id, "hi", mode="agent", approve=approve)]
    assert asked == ["run_command"]


async def test_kernel_unknown_tools_profile_falls_back_to_native(tmp_path, capsys):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm, native_tools="typo")
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    names = {t["function"]["name"] for t in llm.seen_tools[0]}
    assert names == set(ALL_TOOL_NAMES)  # 拼错就退回 native，不能变成"没有工具"


# ---------------------------------------------------------------------------
# 6) ACP 映射：kind 差集与审批翻译
# ---------------------------------------------------------------------------


def test_acp_kind_covers_every_kernel_toolkind():
    """回归：kernel 产出的 kind 必须全部有 ACP 译法。

    漏映射的后果不是报错，而是 **pydantic 校验失败后被客户端静默吞掉** ——
    现象是工具卡片永远停在"运行中"，而且两边日志都干干净净。
    所以这里直接从 ToolCallStart 的 Literal 里取全部取值来比，而不是手写一份清单：
    以后往 contracts 加了新 kind，这个测试会立刻红。
    """
    import typing

    allowed = set(typing.get_args(_TCS.model_fields["kind"].annotation))
    assert allowed, "取不到 ToolKind 取值，说明 contracts 里改了声明方式"
    assert allowed <= set(_ACP_KIND), f"这些 kind 没有 ACP 译法：{allowed - set(_ACP_KIND)}"
    assert set(_ACP_KIND.values()) <= {
        "read",
        "edit",
        "delete",
        "move",
        "search",
        "execute",
        "think",
        "fetch",
        "switch_mode",
        "other",
    }


class _FakeConn:
    """记录 session_update 与 request_permission 的假连接。"""

    def __init__(self, outcome: str = "selected", option_id: str = "allow_once") -> None:
        from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

        self.updates: list[tuple[str, object]] = []
        self.permission_calls: list[dict] = []
        # 注意 outcome 是**必填**的 Literal（没有默认值），构造时必须显式给 ——
        # 这和客户端回传的 JSON 形状一致（{"outcome":"selected","optionId":"..."}）。
        self._response = RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=option_id)
            if outcome == "selected"
            else DeniedOutcome(outcome="cancelled")
        )

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))

    async def request_permission(self, session_id, tool_call, options):
        self.permission_calls.append(
            {"session_id": session_id, "tool_call": tool_call, "options": options}
        )
        return self._response


def _agent_with(conn, tmp_path, script: list[dict]):
    from agentd.boot import Settings, build_kernel, build_store  # noqa: F401
    from agentd.kernel.llm import ScriptLLM
    from agentd.transports.acp_stdio import AgentdAcpAgent

    kernel = AgentKernel(llm=ScriptLLM(script))
    agent = AgentdAcpAgent(kernel)
    agent._conn = conn  # noqa: SLF001 - 这是传输层自己的字段，测试直接注入
    return agent, kernel


async def test_acp_request_permission_allow_once(tmp_path):
    conn = _FakeConn(outcome="selected", option_id="allow_once")
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001

    ok = await approve(
        ApprovalRequest(call_id="c1", tool="run_command", title="run_command", kind="execute", detail="command: ls")
    )
    assert ok is True
    call = conn.permission_calls[0]
    assert call["session_id"] == "sess_1"
    assert call["tool_call"].kind == "execute"
    assert call["tool_call"].raw_input == {"detail": "command: ls"}
    # option_id 是我们自己定义的字符串，客户端只原样回传
    assert {o.option_id for o in call["options"]} == {"allow_once", "allow_session", "reject"}


async def test_acp_request_permission_cancelled_means_denied(tmp_path):
    conn = _FakeConn(outcome="cancelled")
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001

    assert await approve(ApprovalRequest(call_id="c1", tool="edit", title="edit", kind="edit")) is False


async def test_acp_allow_session_remembers_and_stops_asking(tmp_path):
    """选了"本会话总是允许"之后不能每次都弹 —— 审批疲劳比没有审批更危险。"""
    conn = _FakeConn(outcome="selected", option_id="allow_session")
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001
    req = ApprovalRequest(call_id="c1", tool="write_file", title="write_file", kind="edit")

    assert await approve(req) is True
    assert await approve(req) is True
    assert len(conn.permission_calls) == 1  # 只问了一次

    # 另一个会话不继承这个记忆
    other = agent._make_approver("sess_2")  # noqa: SLF001
    assert await other(req) is True
    assert len(conn.permission_calls) == 2


async def test_acp_permission_failure_is_denied(tmp_path):
    """客户端没实现 session/request_permission 时不能放行。"""
    conn = _FakeConn()
    conn.request_permission = _raise  # type: ignore[assignment]
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001

    assert await approve(ApprovalRequest(call_id="c1", tool="edit", title="edit", kind="edit")) is False


async def _raise(*_a, **_kw):
    raise RuntimeError("client does not support session/request_permission")


async def test_acp_prompt_end_to_end_with_native_tool(tmp_path):
    """走真的 prompt()：工具 start 通知的 kind 必须是 ACP 合法值，且 stdout 干净。

    这里直接用假 conn，不起子进程 —— 子进程版本在 test_acp.py 里用 script 后端覆盖。
    """
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    conn = _FakeConn()
    agent, _ = _agent_with(
        conn,
        tmp_path,
        [
            {"tool_calls": [{"name": "glob", "arguments": {"pattern": "*.txt"}}]},
            {"text": "找到了"},
        ],
    )
    new_session = await agent.new_session(cwd=str(tmp_path))
    session_id = new_session.session_id  # new_session 返回的是响应模型，不是裸 id

    from acp.schema import TextContentBlock

    resp = await agent.prompt(session_id, [TextContentBlock(type="text", text="列出 txt")])
    assert resp.stop_reason == "end_turn"

    kinds = [getattr(u, "kind", None) for _, u in conn.updates]
    assert "search" in kinds  # glob ⇒ search，而不是以前写死的 execute
    assert all(k is None or k in _ACP_KIND.values() for k in kinds)

    outputs = [getattr(u, "raw_output", None) for _, u in conn.updates]
    assert any(o and "hello.txt" in o for o in outputs)


# ---------------------------------------------------------------------------
# 7) 联网工具（web_search / web_fetch）
# ---------------------------------------------------------------------------
# 真正发 HTTP 的那一跳不在这里测 —— 它依赖外网，进 CI 只会随机变红。
# 这里只测"解析 / 编码 / 错误路径 / 接线"，全是纯函数或本地必然失败的网络地址；
# 真实的联网验证在 tests/test_web_tools_live.py（要显式开 AGENTD_LIVE_WEB=1）。


# 照 Bing 结果页的形状手写的最小样本（b_algo 块 + h2>a + p + sb_count）
_FAKE_BING_PAGE = """
<html><body><ol id="b_results">
<li class="b_algo"><h2><a href="https://a.example/1">第一条 &amp; 标题</a></h2>
  <p>第一段摘要<br>带换行</p></li>
<li class="b_algo"><h2><a href="https://b.example/2">第二条</a></h2>
  <div class="b_caption"><p>第二段摘要</p></div></li>
<li class="b_algo"><h2><a href="https://c.example/3">第三条</a></h2></li>
</ol><span class="sb_count">约 1,234 条结果</span></body></html>
"""


def test_parse_bing_extracts_title_url_and_snippet():
    results, hint = _parse_bing(_FAKE_BING_PAGE, 10)
    assert [r["title"] for r in results] == ["第一条 & 标题", "第二条", "第三条"]
    assert [r["url"] for r in results] == [
        "https://a.example/1",
        "https://b.example/2",
        "https://c.example/3",
    ]
    # 实体要还原、标签要剥掉、换行要折成空格
    assert results[0]["snippet"] == "第一段摘要带换行"
    assert results[1]["snippet"] == "第二段摘要"
    assert results[2]["snippet"] == ""  # 没有 <p> 就是空摘要，不是 None
    assert hint == "约 1,234 条结果"


def test_parse_bing_honors_limit():
    assert len(_parse_bing(_FAKE_BING_PAGE, 2)[0]) == 2
    assert len(_parse_bing(_FAKE_BING_PAGE, 1)[0]) == 1


def test_parse_bing_on_empty_or_garbage_page():
    """解析不出来必须返回空，让上层报错 —— 不能编。"""
    assert _parse_bing("", 5) == ([], "")
    assert _parse_bing("<html><body>什么都没有</body></html>", 5) == ([], "")


def test_unwrap_bing_redirect_url():
    """Bing 的 /ck/a?u=a1<base64url> 包装要能还原成目标站。"""
    import base64

    target = "https://example.com/a/b?x=1&y=2"
    payload = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    wrapped = f"https://www.bing.com/ck/a?u=a1{payload}&ntb=1"
    assert _unwrap_bing_url(wrapped) == target


def test_unwrap_bing_leaves_direct_urls_alone():
    for url in ("https://example.com/x", "http://a.b/c?d=1"):
        assert _unwrap_bing_url(url) == url


def test_unwrap_bing_bad_payload_returns_original():
    """解不开就原样返回，绝不能抛异常、更不能安静地解出空串。

    b64 解码会**宽容地**忽略非字母表字符，所以垃圾输入有两种坏法：
    解成空串（`u=a1!!!!`），或者因为非 ASCII 直接抛 ValueError。
    两条路都必须退回原值 —— 否则结果里会出现空 URL。
    """
    for bad in (
        "https://www.bing.com/ck/a?u=a1!!!!",
        "https://www.bing.com/ck/a?u=a1这不是base64!!!",
        "https://www.bing.com/ck/a",
    ):
        assert _unwrap_bing_url(bad) == bad
    # 解出来不是 URL 的（比如刚好是别的 base64 文本）也要退回原值
    import base64

    not_a_url = base64.urlsafe_b64encode(b"just some text").decode().rstrip("=")
    wrapped = f"https://www.bing.com/ck/a?u=a1{not_a_url}"
    assert _unwrap_bing_url(wrapped) == wrapped


def test_html_to_text_strips_script_and_style():
    page = (
        "<html><head><style>body{color:red}</style>"
        "<script>var secret=1;</script></head>"
        "<body><p>正文</p></body></html>"
    )
    text = _html_to_text(page)
    assert "正文" in text
    assert "secret" not in text and "color:red" not in text


def test_html_to_text_unescapes_and_keeps_block_breaks():
    text = _html_to_text("<div>a &lt;b&gt;</div><div>c</div>")
    # 实体必须还原，否则模型读到的是 "a &lt;b&gt;" 而不是 "a <b>"
    assert "a <b>" in text
    # 每个块级标签各折成一个换行；相邻的两个块各贡献一个，
    # 于是两块之间留出一个空行 —— 段落间空一行正是想要的读感。
    assert text.splitlines() == ["a <b>", "", "c"]


def test_html_to_text_collapses_runs_of_blank_lines():
    text = _html_to_text("<p>a</p><p></p><p></p><p></p><p>b</p>")
    assert text == "a\n\nb"


def test_looks_textual_is_deliberately_lenient():
    assert _looks_textual("text/html; charset=utf-8")
    assert _looks_textual("application/json")
    assert _looks_textual("")  # 没声明就当文本试一次
    assert _looks_textual("application/octet-stream") is False
    assert _looks_textual("application/pdf") is False


def test_decode_honors_declared_charset_and_falls_back():
    assert _decode("中文".encode("gbk"), "text/html; charset=gbk") == "中文"
    # 声明了一个 Python 不认识的编码名，不能炸
    assert _decode(b"abc", "text/html; charset=no-such-charset") == "abc"


def test_html_to_text_falls_back_to_title_when_body_is_js_only():
    """正文全由 JS 渲染的页面（头条就是）：剥完标签 body 是空的。

    这时 `<title>` 是仅剩的可读信息，必须捞出来 —— 否则 web_fetch 只能空手而归
    （用户截图里那三张红色失败卡）。
    """
    page = (
        "<html><head><title>成都到北京 3 月新增多条直飞航线</title>"
        '<meta charset="UTF-8"></head>'
        "<body></body><script>var glb=1;/* 一大坨混淆 JS */</script></html>"
    )
    assert _html_to_text(page) == "成都到北京 3 月新增多条直飞航线"


def test_html_to_text_falls_back_to_meta_description():
    """没有 <title> 时退到 meta description / og:description。"""
    page = (
        "<html><head>"
        '<meta name="description" content="2024 夏秋航季 成都出发新航线加密">'
        "</head><body></body></html>"
    )
    assert _html_to_text(page) == "2024 夏秋航季 成都出发新航线加密"

    # og:description 同样认；content 写在 name 之前的属性顺序也要能解
    page2 = (
        "<html><head>"
        '<meta content="首都机场航班量回升" property="og:description">'
        "</head><body></body></html>"
    )
    assert _html_to_text(page2) == "首都机场航班量回升"


def test_html_to_text_uses_head_summary_when_body_is_empty():
    """同时有 title 和描述时，两者都要出现在兜底结果里（去重拼接）。"""
    page = (
        "<html><head><title>东航加密成都-北京</title>"
        '<meta name="description" content="每日新增两班"></head>'
        "<body></body></html>"
    )
    out = _html_to_text(page)
    assert "东航加密成都-北京" in out and "每日新增两班" in out


def test_html_to_text_prefers_real_body_over_head_summary():
    """有正文时必须用正文，兜底只在正文为空时才触发。"""
    page = (
        "<html><head><title>标题不该出现</title></head>"
        "<body><p>真正的正文段落</p></body></html>"
    )
    out = _html_to_text(page)
    assert out == "真正的正文段落"
    assert "标题不该出现" not in out


def test_head_summary_reflects_what_web_fetch_fell_back_to():
    """`fell_back` 出参必须和 `_html_to_text` 的兜底分支判断一致。

    web_fetch 靠它决定要不要给结果加"内容不完整"的说明 —— 判反了就会把
    正常正文误标成摘要，或者把只有摘要的页面当全文给出去。
    """
    empty_body = "<html><head><title>只有标题</title></head><body></body></html>"
    real_body = "<html><head><title>标题</title></head><body><p>正文</p></body></html>"

    flag = [False]
    out = _html_to_text(empty_body, fell_back=flag)
    assert flag[0] is True and out == "只有标题"

    flag = [False]
    out = _html_to_text(real_body, fell_back=flag)
    assert flag[0] is False and out == "正文"

    # 不传 out-param 时行为不变（其它调用方无需关心）
    assert _html_to_text(real_body) == "正文"


def test_web_search_requires_query_without_touching_network():
    tb = box(Path.cwd())
    out = asyncio.run(tb.call("web_search", json.dumps({"query": "   "})))
    assert out.startswith("[错误]") and "query" in out


def test_web_fetch_rejects_non_http_scheme():
    tb = box(Path.cwd())
    out = asyncio.run(tb.call("web_fetch", json.dumps({"url": "ftp://example.com/x"})))
    assert out.startswith("[错误]") and "http" in out
    assert asyncio.run(tb.call("web_fetch", "{}")).startswith("[错误]")


def _bypass_sandbox_proxy(monkeypatch):
    """摘掉代理环境变量，让请求直达环回地址。

    本机（以及任何带 HTTP_PROXY 的环境）会把出网流量统统导给本地代理，
    连 http://127.0.0.1:9 也会被代理成 HTTP 502 —— 那样测到的是
    "服务器返回错误码"分支，而不是我们想验的 "根本连不上" 分支。
    清掉代理后内核会立刻拒绝连接（ConnectError），归入 httpx.HTTPError。
    """
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")


def test_web_search_reports_connection_failure(monkeypatch, no_live_credentials):
    """连不上时必须回 [错误] 前缀（AgentMode 靠它把卡片标成 failed），而不是抛异常。

    指向 127.0.0.1:9 是刻意的：本地必然立刻拒绝，不产生任何外网流量。
    三个后端都要指过去 —— 主后端挂了会去试备用，只堵一个的话这条测试会真连外网，
    既离线不了、结果也随网络波动。智谱那层光堵端点不够：仓库 .env 里有真 key，
    必须靠 no_live_credentials 把 key 剥干净，否则真 key 在场时智谱会直接返回结果。
    """
    import agentd.kernel.tools as tools_mod

    monkeypatch.setenv("AGENTD_ZHIPU_SEARCH_ENDPOINT", "http://127.0.0.1:9/zhipu")
    _bypass_sandbox_proxy(monkeypatch)
    monkeypatch.setattr(tools_mod, "_SEARCH_ENDPOINT", "http://127.0.0.1:9/search")
    monkeypatch.setattr(tools_mod, "_SOGOU_ENDPOINT", "http://127.0.0.1:9/web")
    tb = box(Path.cwd())
    out = asyncio.run(tb.call("web_search", json.dumps({"query": "x"})))
    assert out.startswith("[错误]"), out
    assert "web_search 失败" in out
    assert "搜狗也不可用" in out  # 兜底链路确实被走到了


def test_web_fetch_reports_connection_failure(monkeypatch):
    _bypass_sandbox_proxy(monkeypatch)
    tb = box(Path.cwd())
    out = asyncio.run(tb.call("web_fetch", json.dumps({"url": "http://127.0.0.1:9/x"})))
    assert out.startswith("[错误]") and "抓取" in out


def test_is_tls_error_detects_certificate_failure_through_the_wrapper():
    """证书错的判据：httpx 把它包在 ConnectError 里，原始类型挂在 __cause__ 上。

    只看最外层类型会漏判 —— 那样就不忽略证书重试，用户会看到"抓取失败"红卡，
    而这本来是个可以绕过的站点配置问题（实测 sc.people.com.cn 就是）。
    """
    import ssl

    inner = ssl.SSLCertVerificationError(
        1, "certificate verify failed: Hostname mismatch, certificate is not "
           "valid for 'sc.people.com.cn'. (_ssl.c:1032)"
    )
    wrapped = __import__("httpx").ConnectError("certificate verify failed")
    wrapped.__cause__ = inner
    assert _is_tls_error(wrapped) is True
    # 普通连接失败不能被当成证书问题（否则会对每一条失败链接都放开校验）
    plain = __import__("httpx").ConnectError("Connection refused")
    assert _is_tls_error(plain) is False


def test_web_fetch_retries_without_verification_on_cert_mismatch(monkeypatch):
    """证书不过 → 忽略证书重抓一次，并在结果里明说"证书未校验"。

    这是截图里第二/三类红卡的真实成因：站点证书与域名不匹配，严格校验直接连不上。
    """
    import agentd.kernel.tools as tools_mod

    calls: list[bool] = []

    async def fake_fetch_once(url, *, verify):
        calls.append(verify)
        if verify:
            import ssl

            inner = ssl.SSLCertVerificationError(
                1, "certificate verify failed: Hostname mismatch"
            )
            exc = __import__("httpx").ConnectError("certificate verify failed")
            exc.__cause__ = inner
            raise exc
        return 200, url, "text/html; charset=utf-8", "<html><body><p>正文</p></body></html>".encode()

    monkeypatch.setattr(tools_mod, "_fetch_once", fake_fetch_once)
    tb = box(Path.cwd())
    out = asyncio.run(tb.call("web_fetch", json.dumps({"url": "https://x.example/a"})))

    assert calls == [True, False], "必须先严格校验、失败后再放开重试"
    assert not out.startswith("[错误]")
    assert "证书校验未通过" in out
    assert "正文" in out


def test_web_fetch_does_not_relax_verification_for_ordinary_failures(monkeypatch):
    """普通连接失败（非证书问题）不许放开校验证书 —— 那是安全边界，不能为了
    "能抓"就默认降级。"""
    import agentd.kernel.tools as tools_mod

    calls: list[bool] = []

    async def fake_fetch_once(url, *, verify):
        calls.append(verify)
        raise __import__("httpx").ConnectError("Connection refused")

    monkeypatch.setattr(tools_mod, "_fetch_once", fake_fetch_once)
    tb = box(Path.cwd())
    out = asyncio.run(tb.call("web_fetch", json.dumps({"url": "https://x.example/a"})))

    assert calls == [True], "只试一次，不重试"
    assert out.startswith("[错误]") and "抓取" in out


def test_web_fetch_marks_js_only_pages_as_unreadable_with_a_next_step(monkeypatch):
    """整页空壳（头条那种）→ 明确告诉模型"这条路没用，换来源"，别反复抓。

    截图里三张红卡 = 模型对同一类空壳页连抓三次。错误文案必须把"下一步"写进去。
    """
    import agentd.kernel.tools as tools_mod

    async def fake_fetch_once(url, *, verify):
        return 200, url, "text/html", b"<html><head></head><body></body></html>"

    monkeypatch.setattr(tools_mod, "_fetch_once", fake_fetch_once)
    tb = box(Path.cwd())
    out = asyncio.run(tb.call("web_fetch", json.dumps({"url": "https://x.example/a"})))

    assert out.startswith("[错误]")
    assert "没有可读正文" in out
    assert "web_search" in out and "不要重试同一个链接" in out


def test_search_endpoint_is_overridable_by_env(monkeypatch):
    """AGENTD_SEARCH_ENDPOINT 能换搜索后端。

    跨仓库联网 e2e（native_tools_e2e.py --scenario web）就是靠它把后端指到
    本地假 Bing 上，从而离线可复现；顺带也是 cn.bing.com 不通时的逃生门。
    """
    import agentd.kernel.tools as tools_mod

    monkeypatch.delenv("AGENTD_SEARCH_ENDPOINT", raising=False)
    assert tools_mod._search_endpoint() == tools_mod._SEARCH_ENDPOINT

    monkeypatch.setenv("AGENTD_SEARCH_ENDPOINT", "http://127.0.0.1:1234/search")
    assert tools_mod._search_endpoint() == "http://127.0.0.1:1234/search"

    # 空串当"没设"处理，否则端点会变成空 URL、报一个看不懂的错
    monkeypatch.setenv("AGENTD_SEARCH_ENDPOINT", "")
    assert tools_mod._search_endpoint() == tools_mod._SEARCH_ENDPOINT

    # 搜狗备用后端同款覆盖能力（离线单测要把两个后端都指走，见上面的连接失败测试）
    monkeypatch.delenv("AGENTD_SOGOU_ENDPOINT", raising=False)
    assert tools_mod._sogou_endpoint() == tools_mod._SOGOU_ENDPOINT
    monkeypatch.setenv("AGENTD_SOGOU_ENDPOINT", "http://127.0.0.1:1234/web")
    assert tools_mod._sogou_endpoint() == "http://127.0.0.1:1234/web"
    monkeypatch.setenv("AGENTD_SOGOU_ENDPOINT", "")
    assert tools_mod._sogou_endpoint() == tools_mod._SOGOU_ENDPOINT


def test_web_tools_are_registered_readonly():
    """联网工具是只读的：kind 分别是 search / fetch，且不要求审批。"""
    tb = box(Path.cwd())
    assert tb.kind_of("web_search") == "search"
    assert tb.kind_of("web_fetch") == "fetch"
    assert tb.requires_permission("web_search") is False
    assert tb.requires_permission("web_fetch") is False
    # 两个 kind 都必须在 ACP 的合法值里，否则客户端映射会落到 other
    assert tb.kind_of("web_search") in _ACP_KIND.values()
    assert tb.kind_of("web_fetch") in _ACP_KIND.values()


def test_fetch_kind_is_treated_as_read_only_by_approval():
    """fetch 也算只读：否则 AGENTD_TOOLS_APPROVE=all 会把"读个网页"也变成一路点允许。"""
    for policy in ("native", "all", "none"):
        assert (
            needs_approval(requires=False, kind="fetch", destructive=False, policy=policy)
            is False
        )


def test_web_tools_are_absent_from_read_only_profile():
    """read_only 档刻意不含联网工具：这个档的语义是"纯本地只读"，不带出网。"""
    names = NativeToolbox(cwd=Path.cwd(), profile="read_only").names
    assert "web_search" not in names and "web_fetch" not in names
    assert "web_search" in NativeToolbox(cwd=Path.cwd(), profile="native").names
