"""Windows Job object：把 run_command 的整棵进程树圈起来，超时一锅端。

背景（2026-10-14 事故复盘，见 ForgeAgent-GUI「总是允许还是卡住」事件）：

`create_subprocess_shell` 起的 cmd.exe 里再派生的孙进程如果超时不退
（这台机器上 MSYS 的 pwd.exe 在 agentd 上下文里就会挂起），老代码只做
`proc.kill()` —— 那只是 TerminateProcess(cmd.exe)，孙进程变孤儿继续持有
stdout/stderr 管道写端。而 asyncio 在 Windows 上的 `proc.wait()` **不是
「进程退出就返回」**：base_subprocess.BaseSubprocessTransport._try_finish /
_call_connection_lost 表明，waiters 只在「所有管道 transport 断开（EOF）」
时才被唤醒。孙子持写端 → 管道永不 EOF → `await proc.wait()` 无限期挂死，
整个 asyncio 循环被拖住，直到外部取消（GUI 的 900s session/cancel）才把
这一轮捡回来。实测：plain 模式挂 30s+ 不返回；关 Job 后 0s 返回。

解法：spawn 后立刻把 shell 关进带 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 的
Job。cmd 派生的子孙自动继承 Job 成员资格，超时/取消时 TerminateJobObject
一锅端 + 关句柄，管道全部 EOF，wait() 即刻返回（实测 0s）。

边界与降级：
- 非 Windows 平台：所有入口 no-op（attach 返回 None），调用方走老路径。
  POSIX 上 wait() 只等直接子进程，本来就没有这个挂死问题。
- Job 创建/指派失败（进程已退出、已在别的 Job 里且系统不支持嵌套等）：
  attach 返回 None，静默降级，不影响本次执行；调用方应对 wait() 做有界等待。
- 正常结束时 close() 会杀掉仍存活的成员（KILL_ON_JOB_CLOSE 语义）——
  这是有意的：run_command 的契约是「这条命令连同它的子孙随调用结束而终结」，
  不留孤儿。agentd 没有后台任务机制，不需要 detached 存活语义。
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # ---- winnt.h / jobapi.h 常量 ----
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9      # JOBOBJECTINFOCLASS
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _STATUS_SUCCESS = 0

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            # JOBOBJECT_IO_COUNTERS = 6 × ULONGLONG = 48 字节（x64）
            ("IoInfo", ctypes.c_byte * 48),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _create_kill_on_close_job() -> int:
    """建一个带 KILL_ON_JOB_CLOSE 的 Job，返回句柄（调用方负责 CloseHandle）。"""
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")
    try:
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            raise OSError(f"SetInformationJobObject failed: {ctypes.get_last_error()}")
    except BaseException:
        _kernel32.CloseHandle(job)
        raise
    return job


def _assign_pid(job: int, pid: int) -> None:
    """把 pid 指的进程关进 Job。权限只取 AssignProcessToJobObject 必需的最小集。"""
    handle = _kernel32.OpenProcess(
        _PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid
    )
    if not handle:
        raise OSError(f"OpenProcess({pid}) failed: {ctypes.get_last_error()}")
    try:
        ok = _kernel32.AssignProcessToJobObject(job, handle)
        if not ok:
            raise OSError(f"AssignProcessToJobObject failed: {ctypes.get_last_error()}")
    finally:
        _kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """探测 pid 是否还活着（测试用；只依赖同步查询，不碰事件循环）。"""
    if not _IS_WINDOWS or not pid:
        return False
    # SYNCHRONIZE 足够做 WaitForSingleObject 探测；拿不到句柄视为已退出
    handle = _kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        # WAIT_TIMEOUT=0x102：还没信号（活着）；WAIT_OBJECT_0=0：已退出
        return _kernel32.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        _kernel32.CloseHandle(handle)


class ProcessTree:
    """把单个子进程（连同它之后派生的所有后代）圈进一个 Job。

    用法::

        tree = ProcessTree.attach(proc.pid)   # None = 没有 Job 保护
        ...
        tree.kill_tree()   # 超时/取消：显式 TerminateJobObject + 关句柄
        tree.close()       # 幂等收尾；关最后一个句柄也会杀掉残留成员

    两步都幂等；失败静默（Job 保护是增强项，不是正确性依赖）。
    """

    __slots__ = ("_job",)

    def __init__(self, job_handle: int) -> None:
        self._job = job_handle

    @classmethod
    def attach(cls, pid: int) -> "ProcessTree | None":
        """spawn 之后立刻调用；非 Windows 或失败时返回 None（降级）。"""
        if not _IS_WINDOWS or not pid:
            return None
        try:
            job = _create_kill_on_close_job()
        except OSError:
            return None
        try:
            _assign_pid(job, pid)
        except OSError:
            # 典型竞态：命令瞬间结束，pid 已经不存在 —— 没东西可杀，直接放行
            _kernel32.CloseHandle(job)
            return None
        return cls(job)

    def kill_tree(self) -> None:
        """TerminateJobObject：Job 里所有成员（shell + 孙进程）全杀。"""
        job, self._job = self._job, None
        if not job:
            return
        try:
            _kernel32.TerminateJobObject(job, 1)
        finally:
            _kernel32.CloseHandle(job)

    def close(self) -> None:
        """关句柄。KILL_ON_JOB_CLOSE 下，残留成员会随之被杀（有意的语义）。"""
        job, self._job = self._job, None
        if job:
            _kernel32.CloseHandle(job)
