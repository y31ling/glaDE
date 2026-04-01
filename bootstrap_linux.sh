#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERR ]${NC} $*"; }

# ── Install mode (set by choose_install_mode) ─────────────
USE_VENV=1   # 1 = virtual environment (default), 0 = global/system Python

choose_install_mode() {
  echo
  echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
  echo    "  GLADE Installation Mode / 安装模式选择"
  echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
  echo    "  [1] Virtual environment"
  echo    "      Isolated in .venv/ — does not affect system Python"
  echo    "      隔离在 .venv/ 中，不影响系统 Python"
  echo
  echo    "  [2] Global / System Python"
  echo    "      Installs packages into the system Python directly"
  echo    "      直接安装到系统 Python，运行脚本时无需 source env.sh"
  echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
  read -rp "  Choose [1/2] (default: 1): " _choice
  case "${_choice}" in
    2) USE_VENV=0; info "Mode: global system Python install." ;;
    *) USE_VENV=1; info "Mode: virtual environment (.venv/)." ;;
  esac
  echo
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GLAFIC_SRC_DIR="${SCRIPT_DIR}/glafic2"
DEPS_SRC_DIR="${SCRIPT_DIR}/deps/src"
DEPS_PREFIX="${SCRIPT_DIR}/deps/install"
LIB_DIR="${DEPS_PREFIX}/lib"
VENV_DIR="${SCRIPT_DIR}/.venv"

CFITSIO_VERSION="4.6.2"
FFTW_VERSION="3.3.10"
GSL_VERSION="2.8"

install_system_packages() {
  local missing=0
  local packages=(
    build-essential
    pkg-config
    python3
    python3-dev
    python3-venv
    python3-pip
    wget
    curl
    tar
    git
    libcurl4-openssl-dev
    zlib1g-dev
  )

  if ! command -v apt-get >/dev/null 2>&1; then
    warn "未检测到 apt-get，请手动安装系统依赖后重试。"
    return
  fi

  for p in "${packages[@]}"; do
    if ! dpkg -s "${p}" >/dev/null 2>&1; then
      missing=1
      break
    fi
  done

  if [[ "${missing}" -eq 0 ]]; then
    info "系统依赖已满足。"
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    err "缺少 sudo，无法自动安装系统依赖。请手动安装后重试。"
    exit 1
  fi

  info "安装系统依赖..."
  sudo apt-get update
  sudo apt-get install -y "${packages[@]}"
}

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ ! -f "${out}" ]]; then
    info "下载 $(basename "${out}")"
    wget -O "${out}" "${url}"
  fi
}

build_cfitsio() {
  mkdir -p "${DEPS_SRC_DIR}"
  cd "${DEPS_SRC_DIR}"
  local dir="cfitsio-${CFITSIO_VERSION}"
  local tarball="${dir}.tar.gz"
  download_if_missing \
    "https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/${tarball}" \
    "${tarball}"
  [[ -d "${dir}" ]] || tar -xzf "${tarball}"
  cd "${dir}"
  if [[ ! -f "${LIB_DIR}/libcfitsio.so" && ! -f "${LIB_DIR}/libcfitsio.a" ]]; then
    info "编译 CFITSIO ${CFITSIO_VERSION}"
    ./configure --prefix="${DEPS_PREFIX}" --enable-reentrant
    make -j"$(nproc)"
    make install
  else
    info "CFITSIO 已存在，跳过。"
  fi
}

build_fftw() {
  mkdir -p "${DEPS_SRC_DIR}"
  cd "${DEPS_SRC_DIR}"
  local dir="fftw-${FFTW_VERSION}"
  local tarball="${dir}.tar.gz"
  download_if_missing "http://www.fftw.org/${tarball}" "${tarball}"
  [[ -d "${dir}" ]] || tar -xzf "${tarball}"
  cd "${dir}"
  if [[ ! -f "${LIB_DIR}/libfftw3.so" && ! -f "${LIB_DIR}/libfftw3.a" ]]; then
    info "编译 FFTW ${FFTW_VERSION}"
    ./configure --prefix="${DEPS_PREFIX}" --enable-shared --enable-threads
    make -j"$(nproc)"
    make install
  else
    info "FFTW 已存在，跳过。"
  fi
}

build_gsl() {
  mkdir -p "${DEPS_SRC_DIR}"
  cd "${DEPS_SRC_DIR}"
  local dir="gsl-${GSL_VERSION}"
  local tarball="${dir}.tar.gz"
  download_if_missing "https://ftp.gnu.org/gnu/gsl/${tarball}" "${tarball}"
  [[ -d "${dir}" ]] || tar -xzf "${tarball}"
  cd "${dir}"
  if [[ ! -f "${LIB_DIR}/libgsl.so" && ! -f "${LIB_DIR}/libgsl.a" ]]; then
    info "编译 GSL ${GSL_VERSION}"
    ./configure --prefix="${DEPS_PREFIX}"
    make -j"$(nproc)"
    make install
  else
    info "GSL 已存在，跳过。"
  fi
}

write_glafic_makefile() {
  cd "${GLAFIC_SRC_DIR}"
  if [[ -f Makefile && ! -f Makefile.original ]]; then
    cp Makefile Makefile.original
  fi
  cat > Makefile <<EOF
UNAME_S := \$(shell uname -s)
UNAME_M := \$(shell uname -m)

INSTALL_PREFIX = ${DEPS_PREFIX}
LIBPATH = \$(INSTALL_PREFIX)/lib
INCPATH = \$(INSTALL_PREFIX)/include

CC	= gcc
CFLAGS	= -O2 -Wall -fPIC -I\$(INCPATH)
LIBS	= -lm -L\$(LIBPATH) -lcfitsio -lfftw3 -lgsl -lgslcblas -lcurl -lz -Wl,-rpath,\$(LIBPATH)

BIN	= glafic
OBJ_BIN	= glafic.o
OBJS	= call.o ein_tab.o mass.o util.o fits.o \\
	  init.o distance.o gsl_zbrent.o gsl_integration.o \\
	  source.o extend.o point.o opt_extend.o opt_lens.o \\
	  opt_point.o example.o mock.o calcein.o vary.o \\
	  gnfw_tab.o commands.o mcmc.o amoeba_opt.o app_ell.o

LIB	= libglafic.a
AR	= ar
AFLAGS	= r

PY	= python/glafic/glafic.so
OBJ_PY	= python.o
CFLAGS3	= -Wall -shared -L\$(LIBPATH) -Wl,-rpath,\$(LIBPATH)
PY_INC  := \$(shell python3-config --includes)
PY_LDS  := \$(shell python3-config --ldflags --embed)
PY_LIBS = -lm -L\$(LIBPATH) -lcfitsio -lfftw3 -lgsl -lgslcblas -lcurl -lz -Wl,-rpath,\$(LIBPATH)

default: all
all: bin lib python

bin: \$(OBJ_BIN) \$(OBJS)
	\$(CC) \$(CFLAGS) -o \$(BIN) \$(OBJ_BIN) \$(OBJS) \$(LIBS)

lib: \$(OBJS)
	\$(AR) \$(AFLAGS) \$(LIB) \$(OBJS)

python: \$(OBJ_PY) \$(OBJS)
	\$(CC) \$(CFLAGS3) -o \$(PY) \$(OBJ_PY) \$(OBJS) \$(PY_LDS) \$(PY_LIBS)

python.o:python.c glafic.h
	\$(CC) \$(CFLAGS) \$(PY_INC) -c \$< -o \$@

%.o:%.c glafic.h
	\$(CC) \$(CFLAGS) -c \$< -o \$@

clean:
	-rm -f \$(BIN) \$(OBJ_BIN) \$(OBJS) \$(LIB) \$(PY) \$(OBJ_PY) *~ \\#* core*
EOF
}

build_glafic() {
  if [[ ! -d "${GLAFIC_SRC_DIR}" ]]; then
    err "未找到 glafic2 源码目录：${GLAFIC_SRC_DIR}"
    exit 1
  fi
  write_glafic_makefile
  cd "${GLAFIC_SRC_DIR}"
  info "编译 glafic2（二进制 + Python 模块）..."
  make clean || true
  make -j"$(nproc)" all

  if [[ ! -f "${GLAFIC_SRC_DIR}/glafic" ]]; then
    err "glafic 可执行文件编译失败。"
    exit 1
  fi
  if [[ ! -f "${GLAFIC_SRC_DIR}/python/glafic/glafic.so" ]]; then
    err "glafic Python 模块编译失败。"
    exit 1
  fi
}

setup_python_env() {
  if [[ "${USE_VENV}" -eq 1 ]]; then
    info "创建 Python 虚拟环境 (.venv/)..."
    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install -r "${SCRIPT_DIR}/requirements.txt"
  else
    info "全局安装 Python 依赖 (system Python)..."
    # Detect if pip needs --break-system-packages (PEP 668, Ubuntu 23.04+)
    local bsp_flag=""
    if pip3 install --help 2>&1 | grep -q -- "--break-system-packages"; then
      bsp_flag="--break-system-packages"
    fi
    # shellcheck disable=SC2086
    pip3 install ${bsp_flag} --upgrade pip setuptools wheel
    # --ignore-installed: skip distutils-managed packages (e.g. blinker)
    # that pip cannot uninstall; new versions are installed alongside them.
    # shellcheck disable=SC2086
    pip3 install ${bsp_flag} --ignore-installed -r "${SCRIPT_DIR}/requirements.txt"
  fi
}

# ── Register glafic Python module path via .pth file ──────
# This allows `import glafic` to work from ANY Python invocation
# without needing to source env.sh first.
install_glafic_to_python() {
  local glafic_py_dir="${GLAFIC_SRC_DIR}/python"
  local pth_name="glafic_glade.pth"
  local site_pkgs

  if [[ "${USE_VENV}" -eq 1 ]]; then
    site_pkgs="$("${VENV_DIR}/bin/python3" -c \
      "import site; print(site.getsitepackages()[0])")"
  else
    site_pkgs="$(python3 -c \
      "import site; print(site.getsitepackages()[0])")"
  fi

  info "注册 glafic 模块路径到 site-packages..."
  echo "${glafic_py_dir}" > "${site_pkgs}/${pth_name}"
  info "  ${glafic_py_dir}"
  info "  -> ${site_pkgs}/${pth_name}"
}

write_env_script() {
  # Write the common header
  {
    cat <<'ENVEOF'
#!/usr/bin/env bash
set -e
GLADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLAFIC_HOME="${GLADE_ROOT}/glafic2"
GLAFIC_PYTHON_PATH="${GLAFIC_HOME}/python"
GLAFIC_LIB_PATH="${GLADE_ROOT}/deps/install/lib"
ENVEOF

    # Conditionally activate venv
    if [[ "${USE_VENV}" -eq 1 ]]; then
      cat <<'VENVEOF'

if [[ -f "${GLADE_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${GLADE_ROOT}/.venv/bin/activate"
fi
VENVEOF
    fi

    cat <<'ENVEOF'

export GLADE_ROOT
export GLAFIC_HOME
export GLAFIC_PYTHON_PATH
export GLAFIC_LIB_PATH
export PYTHONPATH="${GLAFIC_PYTHON_PATH}:${GLADE_ROOT}/tools:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${GLAFIC_LIB_PATH}:${LD_LIBRARY_PATH:-}"
export PATH="${GLAFIC_HOME}:${PATH}"
ENVEOF
  } > "${SCRIPT_DIR}/env.sh"
  chmod +x "${SCRIPT_DIR}/env.sh"
}

write_run_script() {
  cat > "${SCRIPT_DIR}/run_glade.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
python3 "${SCRIPT_DIR}/main.py" "$@"
EOF
  chmod +x "${SCRIPT_DIR}/run_glade.sh"
}

write_webui_script() {
  cat > "${SCRIPT_DIR}/run_webui.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

GLADE_PORT="${GLADE_PORT:-6017}"
export GLADE_PORT

echo "============================================================"
echo "  GLADE WebUI"
echo "  Open in browser: http://localhost:${GLADE_PORT}"
echo "  Press Ctrl+C to stop"
echo "============================================================"
python3 "${SCRIPT_DIR}/web/app.py"
EOF
  chmod +x "${SCRIPT_DIR}/run_webui.sh"
}

verify_installation() {
  local python_bin
  if [[ "${USE_VENV}" -eq 1 ]]; then
    python_bin="${VENV_DIR}/bin/python3"
  else
    python_bin="python3"
  fi

  info "验证 glafic Python 模块（直接调用，不依赖 env.sh）..."
  "${python_bin}" - <<'PY'
import os, sys
try:
    import glafic
    print("  [OK] glafic import succeeded")
    print("       module:", glafic.__file__)
except ImportError as e:
    print("  [FAIL]", e)
    sys.exit(1)
PY

  info "验证 LD_LIBRARY_PATH（通过 env.sh）..."
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.sh"
  python3 - <<'PY'
import os, sys
try:
    import glafic
    print("  [OK] glafic import via env.sh succeeded")
    print("       GLAFIC_HOME:", os.environ.get("GLAFIC_HOME"))
except ImportError as e:
    print("  [FAIL]", e)
    sys.exit(1)
PY
  info "全部验证通过。"
}

main() {
  info "开始为 Linux 自动搭建 GLADE 环境..."
  choose_install_mode
  install_system_packages

  mkdir -p "${DEPS_SRC_DIR}" "${DEPS_PREFIX}/lib" "${DEPS_PREFIX}/include"
  build_cfitsio
  build_fftw
  build_gsl
  build_glafic
  setup_python_env
  install_glafic_to_python   # register glafic in site-packages (.pth)
  write_env_script
  write_run_script
  write_webui_script
  verify_installation

  info "全部完成。"
  echo
  echo "================================================================"
  if [[ "${USE_VENV}" -eq 1 ]]; then
    echo "  Install mode: virtual environment (.venv/)"
    echo "  To run scripts manually:  source ${SCRIPT_DIR}/env.sh"
  else
    echo "  Install mode: global system Python"
    echo "  glafic is importable from any python3 invocation."
    echo "  Still source env.sh to set LD_LIBRARY_PATH and PATH:"
    echo "    source ${SCRIPT_DIR}/env.sh"
  fi
  echo "================================================================"
  echo
  echo "Next steps:"
  echo "  CLI mode:"
  echo "    1) Edit model_use and parameters in ${SCRIPT_DIR}/main.py"
  echo "    2) Run: ${SCRIPT_DIR}/run_glade.sh"
  echo
  echo "  WebUI mode:"
  echo "    1) Run: ${SCRIPT_DIR}/run_webui.sh"
  echo "    2) Open http://localhost:6017 in your browser"
  echo "    (Set GLADE_PORT=<port> to use a different port)"
  echo
  echo "  Run model scripts directly (global mode or after source env.sh):"
  echo "    python3 legacy/v_pointmass_1_0/version_pointmass_1_0.py"
}

main "$@"
