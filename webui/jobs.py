"""Run execution: spawn a real terminal, auto-type the command, tail its log.

This is the V0.4 anti-concurrency mechanism. Each run gets its own job directory
(``runs/<id>/``) and log file, so concurrent runs never collide. The command is
run inside a fresh OS terminal window (gnome-terminal under WSLg here, with
fallbacks) and tee'd to ``job.log``; the WebUI streams that log to the browser
over SSE.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional


def _pid_alive(pid) -> bool:
    """True if *pid* is a live process. Used to detect a worker that crashed /
    was killed / OOM'd before writing a terminal status (POSIX; the WSL/macOS
    targets are POSIX). Note: pid reuse can briefly yield a false positive, which
    only delays the SSE stream's terminal detection — never a wrong result."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True            # exists but owned by another user (not expected here)
    except OSError:
        return False
    return True


@dataclass
class Job:
    id: str
    backend: str
    files: list
    job_dir: str
    log_path: str
    mode: str = "findimage"
    optimizer: Optional[str] = None   # point-source algorithm (findimage/de+mcmc)
    terminal: str = "?"
    pid: Optional[int] = None
    created: str = ""


class JobManager:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.runs_dir = os.path.join(self.root, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- environment + command ----------------------------------------------
    def _python(self) -> str:
        venv = os.path.join(self.root, ".venv", "bin", "python")
        return venv if os.path.exists(venv) else "python3"

    def _env_exports(self) -> str:
        pythonpath = os.pathsep.join([
            os.path.join(self.root, "glafic2", "python"),
            os.path.join(self.root, "Rhongomyniad"),
            os.path.join(self.root, "tools"),
            self.root,
        ])
        ld = os.path.join(self.root, "deps", "install", "lib")
        return (f"export PYTHONPATH={shlex.quote(pythonpath)}${{PYTHONPATH:+:$PYTHONPATH}}; "
                f"export LD_LIBRARY_PATH={shlex.quote(ld)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; ")

    def _build_command(self, job: Job, force: bool) -> str:
        files_arg = " ".join(shlex.quote(f) for f in job.files)
        force_arg = " --force" if force else ""
        opt_arg = (f" --optimizer {shlex.quote(job.optimizer)}"
                   if job.optimizer else "")
        run = (f"{shlex.quote(self._python())} -u "
               f"{shlex.quote(os.path.join('webui', 'runjob.py'))} "
               f"--backend {job.backend} --mode {shlex.quote(job.mode)} "
               f"--out {shlex.quote(job.job_dir)} "
               f"--files {files_arg}{opt_arg}{force_arg}")
        # cd, set env, run, tee to log, keep the window open afterwards
        return (f"cd {shlex.quote(self.root)}; {self._env_exports()}"
                f"{run} 2>&1 | tee {shlex.quote(job.log_path)}; "
                f"echo; echo '[GLADE job finished — press Enter to close]'; read _")

    # -- terminal spawning ---------------------------------------------------
    def _spawn_terminal(self, bash_cmd: str, title: str):
        # macOS: there is no gnome-terminal/x-terminal-emulator. Open Terminal.app
        # on a throwaway .command script (avoids AppleScript quoting issues) so a
        # real window pops up just like the Linux path; the job still tees to
        # job.log, which the WebUI streams over SSE. This branch only runs on
        # macOS, so the Linux behaviour below is untouched.
        if sys.platform == "darwin":
            fd, script_path = tempfile.mkstemp(prefix="glade_", suffix=".command")
            with os.fdopen(fd, "w") as fh:
                fh.write("#!/bin/bash\n")
                fh.write(f"# {title}\n")
                fh.write(bash_cmd + "\n")
            os.chmod(script_path, 0o755)
            p = subprocess.Popen(["open", "-a", "Terminal", script_path])
            return "Terminal.app", p.pid
        if shutil.which("gnome-terminal"):
            p = subprocess.Popen(
                ["gnome-terminal", "--title", title, "--", "bash", "-lc", bash_cmd])
            return "gnome-terminal", p.pid
        if shutil.which("x-terminal-emulator"):
            p = subprocess.Popen(
                ["x-terminal-emulator", "-T", title, "-e", "bash", "-lc", bash_cmd])
            return "x-terminal-emulator", p.pid
        if shutil.which("xterm"):
            p = subprocess.Popen(["xterm", "-T", title, "-e", "bash", "-lc", bash_cmd])
            return "xterm", p.pid
        if shutil.which("tmux"):
            subprocess.Popen(["tmux", "new-session", "-d", "-s", title, bash_cmd])
            return "tmux", None
        # last resort: detached background process (no visible window)
        p = subprocess.Popen(["bash", "-lc", bash_cmd], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "detached", p.pid

    # -- public API ----------------------------------------------------------
    def start(self, backend: str, files: list, mode: str = "findimage",
              force: bool = False, optimizer: Optional[str] = None) -> Job:
        job_id = (datetime.now().strftime("%y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4])
        job_dir = os.path.join(self.runs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        log_path = os.path.join(job_dir, "job.log")
        open(log_path, "w").close()  # create empty log immediately for tailing

        job = Job(id=job_id, backend=backend, files=list(files), job_dir=job_dir,
                  log_path=log_path, mode=mode, optimizer=optimizer,
                  created=datetime.now().isoformat(timespec="seconds"))
        bash_cmd = self._build_command(job, force)
        job.terminal, job.pid = self._spawn_terminal(bash_cmd, f"GLADE {job_id}")
        with self._lock:
            self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def status(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            return {"state": "unknown"}
        sp = os.path.join(job.job_dir, "status.json")
        st = {"state": "starting"}
        if os.path.exists(sp):
            try:
                with open(sp, encoding="utf-8") as fh:
                    st = json.load(fh)
            except (OSError, ValueError):
                # a torn read mid-write: fall back without the worker_pid so the
                # liveness check below is skipped (next poll re-reads cleanly).
                st = {"state": "starting"}
        # A worker that crashed / was killed / ran out of memory before writing a
        # terminal state would otherwise look 'running' forever (the SSE stream
        # then hangs to its idle timeout). If its pid is gone, mark it terminal.
        if st.get("state") in ("starting", "running"):
            wpid = st.get("worker_pid")
            if wpid is not None and not _pid_alive(wpid):
                st = {**st, "state": "interrupted",
                      "error": "the run process exited before finishing "
                               "(crash, kill, or out-of-memory); see the "
                               "terminal window / job.log for details"}
        return st

    def _is_finished(self, job: Job) -> bool:
        return self.status(job.id).get("state") in ("done", "error", "interrupted")

    def tail(self, job_id: str, idle_timeout: float = 1800.0) -> Iterator[str]:
        """Yield log lines as they are written, ending when the job finishes."""
        job = self.jobs.get(job_id)
        if job is None:
            return
        # wait briefly for the log to appear
        waited = 0.0
        while not os.path.exists(job.log_path) and waited < 10:
            time.sleep(0.2)
            waited += 0.2

        last_activity = time.time()
        with open(job.log_path, "r", encoding="utf-8", errors="replace") as fh:
            while True:
                line = fh.readline()
                if line:
                    last_activity = time.time()
                    yield line.rstrip("\n")
                    continue
                # no new data
                if self._is_finished(job):
                    # drain any final bytes then stop
                    rest = fh.read()
                    for ln in rest.splitlines():
                        yield ln
                    return
                if time.time() - last_activity > idle_timeout:
                    yield "[stream timed out — the job may still be running in its terminal]"
                    return
                time.sleep(0.2)
