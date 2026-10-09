"""agentd 一键安装脚本：拉 submodule → 建 venv → 编译安装 → 写路径清单。

以后端仓库为主角（本脚本住在 Agentd/scripts/），前端 GUI 仓库通过
``--gui <路径>`` 传进来，两个包**编进同一个 venv**（``~/.agentd/venv``）
—— 这是 GUI 拉起 agentd 的标准姿势（acp_client 默认 ``sys.executable -m
agentd.server``，同 venv 才能找到 agentd 模块）。

安装完成后在 ``~/.agentd/install.json`` 落一份路径清单（venv 解释器、两个
仓库的绝对路径与 commit、submodule 状态、技能清单）—— 这就是「路径传递」
的落点：任何工具（GUI / 运维脚本 / 人）想知道这套安装长什么样，读它。

用法（在 Agentd 仓库根）::

    python scripts/install.py --gui ../ForgeAgent-GUI
    python scripts/install.py --gui D:/workspace/ForgeAgent-GUI --force

只更新 submodule 与重装，不重建 venv —— 脚本幂等，重复跑就是升级。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import venv
from pathlib import Path

# 单技能正文上限之类不放这 —— 本脚本只管装。技能是 agentd 运行时发现的。

REPO_ROOT = Path(__file__).resolve().parents[1]


class InstallError(RuntimeError):
    """安装失败 —— 带着人话原因，直接打印给用户。"""


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------


def find_git(explicit: str | None = None) -> str:
    """定位 git 可执行文件。这台机器常见坑：git 不在 PATH 上（PortableGit）。"""
    if explicit:
        if Path(explicit).is_file():
            return explicit
        raise InstallError(f"--git 指定的路径不存在：{explicit}")
    import shutil

    hit = shutil.which("git")
    if hit:
        return hit
    # 常见落点的兜底探测（Windows 优先）：
    candidates = [
        Path("C:/Program Files/Git/cmd/git.exe"),
        Path("C:/Program Files (x86)/Git/cmd/git.exe"),
        # WorkBuddy 托管的 PortableGit（versions 下取 cmd/git.exe）
        *sorted(
            Path.home().glob(".workbuddy/binaries/PortableGit/versions/*/cmd/git.exe")
        ),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise InstallError(
        "找不到 git。请用 --git <路径> 指定，或把 git 加进 PATH。\n"
        "拉 submodule 没有它跑不动。"
    )


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    """跑一条命令，回传 stdout。失败抛 InstallError（带完整输出，别吞证据）。"""
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        raise InstallError(
            f"命令失败（exit={proc.returncode}）：{' '.join(cmd)}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# 步骤
# ---------------------------------------------------------------------------


def update_submodules(git: str) -> list[dict]:
    """拉齐全部 submodule（init + 递归），返回它们的状态清单。"""
    run([git, "submodule", "update", "--init", "--recursive"], cwd=REPO_ROOT)
    # git submodule status 每行："<commit> <path> (tag/branch)"，前缀 '-' 表示未 init
    out = run([git, "submodule", "status"], cwd=REPO_ROOT)
    subs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        commit, rest = line[1:].split(" ", 1)
        subs.append({"path": rest.split(" ")[0], "commit": commit, "ready": not line.startswith("-")})
    missing = [s["path"] for s in subs if not s["ready"]]
    if missing:
        raise InstallError(f"submodule 未就绪：{missing}（git 版本过旧或网络失败？）")
    return subs


def validate_gui_path(raw: str | None) -> Path | None:
    """校验前端 GUI 仓库路径：必须有 forgeagent 包与 pyproject。"""
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    if not (p / "pyproject.toml").is_file() or not (p / "forgeagent").is_dir():
        raise InstallError(
            f"--gui 指向的目录不像 ForgeAgent-GUI 仓库（缺 pyproject.toml 或 forgeagent/）：{p}"
        )
    return p


def ensure_venv(home: Path) -> Path:
    """venv 在 ~/.agentd/venv；已存在就复用（升级姿势），缺了才建。"""
    venv_dir = home / "venv"
    py = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if py.is_file():
        return py
    home.mkdir(parents=True, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(venv_dir))
    if not py.is_file():
        raise InstallError(f"venv 建出来了但找不到解释器：{py}")
    return py


def pip_install(venv_py: Path, *targets: Path) -> None:
    """把包以 editable 方式装进 venv（pip 构建过程即字节码编译）。"""
    if not targets:
        return
    cmd = [str(venv_py), "-m", "pip", "install", "--no-input"]
    for t in targets:
        cmd += ["-e", str(t)]
    run(cmd)


def compile_packages(venv_py: Path, *repos: Path) -> int:
    """compileall 把两个仓库的源码预编译成 .pyc（启动更快、顺带暴露语法错）。"""
    total = 0
    for repo in repos:
        pkg = repo / "agentd" if (repo / "agentd").is_dir() else repo / "forgeagent"
        if not pkg.is_dir():
            continue
        out = run([str(venv_py), "-m", "compileall", "-q", str(pkg)], check=False)
        total += 1
    return total


def git_commit(git: str, repo: Path) -> str:
    try:
        return run([git, "rev-parse", "--short", "HEAD"], cwd=repo)
    except InstallError:
        return "unknown"


def verify(venv_py: Path, gui_installed: bool) -> dict:
    """在 venv 里做导入级自检 + 技能发现，结果进 manifest 也打给用户。"""
    code = (
        "import json;"
        "import agentd;"
        "out={'agentd': getattr(agentd,'__version__','0.1.0')};"
        "from agentd.kernel import skills;"
        "out['skills']=sorted(skills.discover());"
        + ("import forgeagent;" if gui_installed else "")
        + "print(json.dumps(out))"
    )
    out = run([str(venv_py), "-c", code])
    return json.loads(out)


def build_manifest(
    *, venv_py: Path, git: str, backend_commit: str, gui: Path | None,
    gui_commit: str | None, submodules: list[dict], verify_out: dict,
) -> dict:
    return {
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "venv_python": str(venv_py),
        "backend": {"repo": str(REPO_ROOT), "commit": backend_commit},
        "gui": ({"repo": str(gui), "commit": gui_commit} if gui else None),
        "submodules": submodules,
        "verify": verify_out,
    }


def write_manifest(home: Path, manifest: dict) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    target = home / "install.json"
    tmp = home / "install.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)  # 原子替换：读的一方永远看不到半截 JSON
    return target


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="agentd 一键安装：submodule → venv → 编译安装 → ~/.agentd/install.json"
    )
    parser.add_argument("--gui", default=None, help="前端 GUI 仓库路径（ForgeAgent-GUI）")
    parser.add_argument("--home", default=None, help="安装根目录，默认 ~/.agentd")
    parser.add_argument("--git", default=None, help="git 可执行文件路径（不在 PATH 上时用）")
    args = parser.parse_args(argv)

    home = Path(args.home).expanduser().resolve() if args.home else Path.home() / ".agentd"
    print(f"[1/6] 仓库根：{REPO_ROOT}")

    try:
        git = find_git(args.git)
        print(f"[2/6] git：{git}")
        print("[3/6] 拉取 submodule …")
        submodules = update_submodules(git)
        for s in submodules:
            print(f"      {s['path']} @ {s['commit'][:8]}")

        gui = validate_gui_path(args.gui)
        print(f"[4/6] venv（~/.agentd/venv）…")
        venv_py = ensure_venv(home)

        targets = [REPO_ROOT] + ([gui] if gui else [])
        names = " + ".join("agentd" if t == REPO_ROOT else "forgeagent-tui" for t in targets)
        print(f"[5/6] 编译安装：{names}（editable，进 {venv_py.parent.parent}）…")
        pip_install(venv_py, *targets)
        compile_packages(venv_py, *targets)

        backend_commit = git_commit(git, REPO_ROOT)
        gui_commit = git_commit(git, gui) if gui else None
        print("[6/6] 自检 …")
        verify_out = verify(venv_py, gui_installed=gui is not None)
        manifest = build_manifest(
            venv_py=venv_py, git=git, backend_commit=backend_commit, gui=gui,
            gui_commit=gui_commit, submodules=submodules, verify_out=verify_out,
        )
        manifest_path = write_manifest(home, manifest)
    except InstallError as exc:
        print(f"安装失败：{exc}", file=sys.stderr)
        return 1

    print()
    print("✅ 安装完成")
    print(f"   路径清单：{manifest_path}")
    print(f"   技能：{', '.join(verify_out.get('skills', [])) or '（无）'}")
    print()
    print("   启动方式：")
    print(f"     GUI：\"{venv_py}\" -m forgeagent.gui")
    print(f"     CLI：\"{venv_py}\" -m agentd cli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
