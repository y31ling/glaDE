#!/usr/bin/env bash
# =============================================================================
#  GLADE updater — pull the latest code from GitHub and refresh the install.
#
#  What it does (menu option 1, "Update"):
#    1. git fetch + fast-forward the current branch from origin (with safe
#       auto-stash of any genuine local edits);
#    2. rebuild the bundled glafic C engine ONLY if its sources changed;
#    3. (re)install / upgrade the Python dependencies;
#    4. regenerate env.sh / run_*.sh launchers if their generator changed;
#    5. offer to add / update the optional GPU backend (PyTorch);
#    6. re-initialize the WebUI feature flag and verify the install.
#
#  Menu option 2 ("Add / update GPU") skips git entirely and only installs
#  PyTorch — for turning a CPU-only install into a GPU one without a rebuild.
#
#  It reuses the exact build functions from bootstrap_linux.sh /
#  bootstrap_macos.sh (sourced, never re-run), so an update builds glade the
#  same way a fresh install does — including a regenerated Makefile if new
#  glafic source files were pulled.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR ]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── pick and source the platform bootstrap (for its build functions) ─────────
OS="$(uname -s)"
case "${OS}" in
  Linux)  BOOT="bootstrap_linux.sh" ;;
  Darwin) BOOT="bootstrap_macos.sh" ;;
  *) err "不支持的系统: ${OS}（仅支持 Linux / macOS）。"; exit 1 ;;
esac
if [[ ! -f "${SCRIPT_DIR}/${BOOT}" ]]; then
  err "未找到 ${BOOT}，无法复用构建逻辑。"; exit 1
fi
# The bootstrap only auto-runs when executed directly; sourcing just loads its
# functions (info/warn/err are redefined identically). Also sets SCRIPT_DIR,
# GLAFIC_SRC_DIR, VENV_DIR, DEPS_* and the default USE_VENV.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/${BOOT}"

# ── detect the existing install mode (venv vs global system Python) ──────────
if [[ -d "${SCRIPT_DIR}/.venv" ]]; then
  USE_VENV=1
else
  USE_VENV=0
fi

OLD_HEAD=""; NEW_HEAD=""

# ── pip helper honouring venv / global mode ──────────────────────────────────
_pip_install() {
  if [[ "${USE_VENV}" -eq 1 ]]; then
    # shellcheck disable=SC1091
    [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"
    pip install "$@"
  else
    local bsp=""
    if pip3 install --help 2>&1 | grep -q -- "--break-system-packages"; then
      bsp="--break-system-packages"
    fi
    # shellcheck disable=SC2086
    pip3 install ${bsp} "$@"
  fi
}

# ── revert only the WebUI flag stamp back to its committed token, so app.js ──
# matches upstream and never blocks a fast-forward. It is re-stamped after the
# update by the reused initialize_prank_flag.
_destamp_flag() {
  local appjs="${SCRIPT_DIR}/webui/static/app.js"
  [[ -f "${appjs}" ]] || return 0
  python3 - "${appjs}" <<'PY' >/dev/null 2>&1 || true
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
new, n = re.subn(r'"glade_af_seen_[0-9A-Za-z\-]+"', '"glade_af_seen"', s, count=1)
if n:
    open(p, "w", encoding="utf-8").write(new)
PY
}

# ── robust git update: fast-forward with safe auto-stash ─────────────────────
git_update() {
  command -v git >/dev/null 2>&1 || { err "未安装 git。"; exit 1; }
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { err "当前目录不是 git 仓库，无法从 GitHub 更新。"; exit 1; }

  local branch upstream
  branch="$(git rev-parse --abbrev-ref HEAD)"
  upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [[ -z "${upstream}" ]]; then
    err "分支 '${branch}' 未设置上游远程分支，无法自动更新。"
    err "可先执行:  git branch --set-upstream-to=origin/${branch} ${branch}"
    exit 1
  fi
  info "当前分支: ${branch}  (跟踪 ${upstream})"

  # Neutralize the two installer-owned tracked files so they never block a
  # fast-forward: the flag stamp (surgically reverted) and the machine-specific
  # generated Makefile (restored to HEAD; regenerated later during rebuild).
  _destamp_flag
  if git ls-files --error-unmatch glafic2/Makefile >/dev/null 2>&1; then
    git checkout -- glafic2/Makefile 2>/dev/null || true
  fi

  info "从 GitHub 获取更新..."
  git fetch --prune origin

  OLD_HEAD="$(git rev-parse HEAD)"
  if git merge --ff-only '@{u}' >/dev/null 2>&1; then
    :
  else
    warn "存在本地改动或无法直接快进，尝试自动暂存本地改动..."
    local stashed=0
    if ! git diff --quiet || ! git diff --cached --quiet || \
       [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
      if git stash push -u -m "glade-update-autostash" >/dev/null 2>&1; then
        stashed=1; info "  本地改动已暂存 (git stash)。"
      fi
    fi
    if git merge --ff-only '@{u}' >/dev/null 2>&1; then
      if [[ ${stashed} -eq 1 ]]; then
        info "  恢复本地改动..."
        if ! git stash pop >/dev/null 2>&1; then
          # the user's local edits overlap the pulled changes -> real conflict.
          # Reset to a CLEAN new version (their edits stay safe in the stash)
          # and stop, rather than build on a tree with conflict markers.
          git reset --hard HEAD >/dev/null 2>&1 || true
          err "本地改动与远程更新存在冲突，更新已停止（未构建）。"
          err "你的改动已安全保存在 git stash 中，未丢失。"
          err "请手动执行:  git stash pop  解决冲突后，再重新运行 ./update.sh"
          exit 1
        fi
      fi
    else
      [[ ${stashed} -eq 1 ]] && git stash pop >/dev/null 2>&1 || true
      err "无法快进更新：本地分支与远程已分叉。"
      err "请手动处理后重试:  git pull --rebase origin ${branch}"
      exit 1
    fi
  fi
  NEW_HEAD="$(git rev-parse HEAD)"

  if [[ "${OLD_HEAD}" == "${NEW_HEAD}" ]]; then
    info "已是最新版本，无新提交。"
  else
    local n; n="$(git rev-list --count "${OLD_HEAD}..${NEW_HEAD}")"
    info "已拉取 ${n} 个新提交："
    git --no-pager log --oneline "${OLD_HEAD}..${NEW_HEAD}" | sed 's/^/      /'
  fi
}

# ── did the given paths change between OLD_HEAD..NEW_HEAD ? ───────────────────
_changed() {
  [[ -z "${OLD_HEAD}" || "${OLD_HEAD}" == "${NEW_HEAD}" ]] && return 1
  git diff --name-only "${OLD_HEAD}" "${NEW_HEAD}" -- "$@" | grep -q .
}

# ── rebuild the native engine only when necessary ────────────────────────────
rebuild_native_if_needed() {
  local need=0
  if _changed glafic2/ "${BOOT}"; then
    if git diff --name-only "${OLD_HEAD}" "${NEW_HEAD}" -- glafic2/ "${BOOT}" \
         | grep -qE '\.(c|h)$|glafic2/Makefile|bootstrap_'; then
      need=1
    fi
  fi
  [[ -f "${GLAFIC_SRC_DIR}/glafic" && -f "${GLAFIC_SRC_DIR}/python/glafic/glafic.so" ]] || need=1

  if [[ ${need} -eq 0 ]]; then
    info "glafic C 引擎无需重建（源码未变化，二进制存在）。"
    return 0
  fi

  info "检测到引擎相关变化，重建 glafic..."
  if [[ "${OS}" == "Darwin" ]]; then
    check_toolchain                       # sets BREW_PREFIX (needed by Makefile)
    # ensure the C dependencies are present (idempotent; brew skips installed)
    install_brew_packages
  else
    # Linux: rebuild the vendored C deps only if they are missing.
    if [[ ! -f "${LIB_DIR}/libgsl.so" && ! -f "${LIB_DIR}/libgsl.a" ]]; then
      warn "未找到已编译的 C 依赖，正在构建 CFITSIO/FFTW/GSL..."
      mkdir -p "${DEPS_SRC_DIR}" "${DEPS_PREFIX}/lib" "${DEPS_PREFIX}/include"
      build_cfitsio; build_fftw; build_gsl
    fi
  fi
  build_glafic                            # regenerates Makefile + make clean && make all
}

# ── refresh Python deps (upgrade only if requirements.txt changed) ───────────
update_python_deps() {
  if [[ ! -f "${SCRIPT_DIR}/requirements.txt" ]]; then
    warn "未找到 requirements.txt，跳过 Python 依赖。"; return 0
  fi
  if _changed requirements.txt; then
    info "requirements.txt 有更新，升级 Python 依赖..."
    _pip_install --upgrade -r "${SCRIPT_DIR}/requirements.txt"
  else
    info "确保 Python 依赖已安装..."
    _pip_install -r "${SCRIPT_DIR}/requirements.txt"
  fi
}

# ── regenerate launchers only if their generator (bootstrap) changed ─────────
refresh_launchers_if_needed() {
  if [[ ! -f "${SCRIPT_DIR}/env.sh" ]] || _changed "${BOOT}"; then
    info "刷新启动脚本 (env.sh / run_glade.sh / run_webui.sh)..."
    [[ "${OS}" == "Darwin" && -z "${BREW_PREFIX:-}" ]] && check_toolchain
    write_env_script; write_run_script; write_webui_script
  fi
}

do_update() {
  info "开始更新 GLADE（${OS} / $([[ ${USE_VENV} -eq 1 ]] && echo venv || echo 'system python')）..."
  git_update
  rebuild_native_if_needed
  update_python_deps
  install_glafic_to_python          # idempotent: refresh the glafic .pth
  refresh_launchers_if_needed
  setup_gpu_optional                # add / update optional GPU (PyTorch)
  initialize_prank_flag             # re-initialize the WebUI feature flag
  verify_installation

  echo
  echo "================================================================"
  info "更新完成。"
  echo "================================================================"
  echo "  运行 WebUI:  ${SCRIPT_DIR}/run_webui.sh   ->  http://localhost:6017"
  echo "  命令行:      ${SCRIPT_DIR}/run_glade.sh"
  echo "  手动环境:    source ${SCRIPT_DIR}/env.sh"
}

do_gpu_only() {
  info "仅添加 / 更新 GPU 支持（不改动代码，不重建 glafic）..."
  setup_gpu_optional
  echo
  info "完成。重新打开 WebUI 后，Clave / FindImage 的 GPU 选项即可用（若 CUDA 可用）。"
}

# ── entry menu ───────────────────────────────────────────────────────────────
choose_action() {
  local action="${1:-}"
  if [[ -z "${action}" ]]; then
    echo
    echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
    echo    "  GLADE Updater / 更新工具"
    echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
    echo    "  [1] Update from GitHub / 从 GitHub 更新"
    echo    "      拉取最新代码 + 重建引擎 + 刷新依赖（结束时可选装 GPU）"
    echo
    echo    "  [2] Add / update GPU support / 添加或更新 GPU 支持"
    echo    "      仅安装 PyTorch（把 CPU 版升级为 GPU 版，不重建）"
    echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
    read -rp "  请选择 / Choose [1/2] (default: 1): " action
  fi
  case "${action}" in
    2|gpu)  do_gpu_only ;;
    *)      do_update ;;
  esac
}

choose_action "${1:-}"
