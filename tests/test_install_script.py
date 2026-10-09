"""scripts/install.py 的纯逻辑测试。

真安装（网络 + venv + pip）不在单测里跑 —— 这里只钉可离线验证的部分：
路径校验、git 定位、submodule 状态解析、清单构建与原子写入。
通过 importlib 从路径加载（scripts/ 不是包）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install.py"
_spec = importlib.util.spec_from_file_location("agentd_install", _SCRIPT)
install = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(install)


# ---------------------------------------------------------------------------
# GUI 路径校验
# ---------------------------------------------------------------------------


def test_validate_gui_path_accepts_real_layout(tmp_path: Path):
    gui = tmp_path / "ForgeAgent-GUI"
    (gui / "forgeagent").mkdir(parents=True)
    (gui / "pyproject.toml").write_text("[project]\nname='forgeagent-tui'\n", encoding="utf-8")
    got = install.validate_gui_path(str(gui))
    assert got is not None and got.is_absolute()


def test_validate_gui_path_rejects_non_repo(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(install.InstallError, match="不像 ForgeAgent-GUI"):
        install.validate_gui_path(str(empty))


def test_validate_gui_path_none_passes_through():
    """不传 --gui 是合法的（纯后端安装）。"""
    assert install.validate_gui_path(None) is None


# ---------------------------------------------------------------------------
# git 定位
# ---------------------------------------------------------------------------


def test_find_git_explicit_missing_path_raises():
    with pytest.raises(install.InstallError, match="--git"):
        install.find_git(str(Path("Z:/nope/git.exe")))


def test_find_git_fallback_candidates(tmp_path: Path, monkeypatch):
    """PATH 上没有 git 时探测常见落点（含 WorkBuddy PortableGit 布局）。"""
    fake = (
        tmp_path / ".workbuddy" / "binaries" / "PortableGit"
        / "versions" / "1.2.0" / "cmd" / "git.exe"
    )
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"")
    monkeypatch.setattr(install.Path, "home", lambda: tmp_path)
    got = install.find_git()
    assert Path(got).is_file()


# ---------------------------------------------------------------------------
# 清单构建与写入
# ---------------------------------------------------------------------------


def test_build_manifest_shape(tmp_path: Path):
    m = install.build_manifest(
        venv_py=tmp_path / "venv" / "python.exe",
        git="git",
        backend_commit="abc1234",
        gui=tmp_path / "gui",
        gui_commit="def5678",
        submodules=[{"path": "vendor/superpowers", "commit": "a" * 40, "ready": True}],
        verify_out={"agentd": "0.1.0", "skills": ["writing-plans"]},
    )
    assert m["backend"]["repo"] == str(install.REPO_ROOT)
    assert m["gui"]["commit"] == "def5678"
    assert m["submodules"][0]["ready"] is True
    assert "installed_at" in m


def test_write_manifest_is_atomic_and_readable(tmp_path: Path):
    manifest = {"venv_python": "x", "backend": {"repo": "r", "commit": "c"}}
    target = install.write_manifest(tmp_path, manifest)
    assert target.name == "install.json"
    assert json.loads(target.read_text(encoding="utf-8")) == manifest
    # 半截产物不落地：tmp 文件已被替换掉
    assert not (tmp_path / "install.json.tmp").exists()


def test_write_manifest_overwrites_previous(tmp_path: Path):
    install.write_manifest(tmp_path, {"v": 1})
    install.write_manifest(tmp_path, {"v": 2})
    assert json.loads((tmp_path / "install.json").read_text(encoding="utf-8")) == {"v": 2}


# ---------------------------------------------------------------------------
# submodule 状态解析（不跑真 git，注入假输出）
# ---------------------------------------------------------------------------


def test_update_submodules_parses_status(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, *, cwd=None, check=True):
        calls.append(cmd)
        if "status" in cmd and "--init" not in cmd:
            return (
                " 683bc88 vendor/anthropic-skills (heads/main)\n"
                " 8ca22db vendor/superpowers (v6.4.2)\n"
            )
        return ""

    monkeypatch.setattr(install, "run", fake_run)
    subs = install.update_submodules("git")
    assert [s["path"] for s in subs] == [
        "vendor/anthropic-skills", "vendor/superpowers",
    ]
    assert all(s["ready"] for s in subs)
    # update 命令必须带 --init --recursive —— 普通 clone 后的第一跑全靠它
    assert any("--init" in c and "--recursive" in c for c in calls)


def test_update_submodules_detects_uninitialized(tmp_path: Path, monkeypatch):
    """status 行首 '-' = 未 init：必须报错而不是静默装出没有技能的系统。"""

    def fake_run(cmd, *, cwd=None, check=True):
        if "status" in cmd and "--init" not in cmd:
            return "- 683bc88 vendor/anthropic-skills (heads/main)\n"
        return ""

    monkeypatch.setattr(install, "run", fake_run)
    with pytest.raises(install.InstallError, match="submodule 未就绪"):
        install.update_submodules("git")
