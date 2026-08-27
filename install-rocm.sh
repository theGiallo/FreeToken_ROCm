#!/usr/bin/env bash
#
# FreeToken engine installer (Linux, AMD ROCm) — source-build, no prebuilt wheels.
#
# Installs the `freetoken` runtime (the `ft` CLI) from source into a managed
# venv. GGUF kernels JIT-compile via hipcc on first use (~1 min). Triton
# fallbacks handle fused MHA / GEMV — no flashinfer or sglang-kernel needed.
#
# Typical use (once from a source checkout):
#   ./install-rocm.sh
#
# Configurable via environment:
#   FREETOKEN_HOME       install root (default: ~/.freetoken); venv at $FREETOKEN_HOME/venv
#   FREETOKEN_PY_VERSION python for the venv (default: 3.12)
#   FREETOKEN_BIN_DIR    where to symlink `ft` (default: ~/.local/bin)
#   FREETOKEN_ENV_DIR    environment.d dir (default: ~/.config/environment.d)
#   FREETOKEN_ROCM_INDEX PyTorch ROCm index URL (default: repo.amd.com)
set -euo pipefail

FT_HOME="${FREETOKEN_HOME:-$HOME/.freetoken}"
VENV="$FT_HOME/venv"
PY_VERSION="${FREETOKEN_PY_VERSION:-3.12}"
BIN_DIR="${FREETOKEN_BIN_DIR:-$HOME/.local/bin}"
ENV_DIR="${FREETOKEN_ENV_DIR:-$HOME/.config/environment.d}"
ROCM_INDEX="${FREETOKEN_ROCM_INDEX:-https://repo.amd.com/rocm/whl-multi-arch/}"

ASSUME_YES="${FREETOKEN_ASSUME_YES:-0}"
for _arg in "$@"; do
  case "$_arg" in
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) printf 'usage: install-rocm.sh [--yes]\n'; exit 0 ;;
  esac
done

if [ -t 1 ]; then
  C_CYAN=$'\033[1;36m'; C_YELLOW=$'\033[1;33m'; C_RED=$'\033[1;31m'; C_GREEN=$'\033[1;32m'; C_RESET=$'\033[0m'
else
  C_CYAN=''; C_YELLOW=''; C_RED=''; C_GREEN=''; C_RESET=''
fi
say()  { printf '%s==>%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf '%s[error]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

# --- 1. uv (bootstrap into the install tree when absent) -------------------
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  if [ "$ASSUME_YES" != 1 ]; then
    if [ -e /dev/tty ]; then
      printf '%suv is not installed. Install it into %s (astral.sh)? [y/N] %s' "$C_YELLOW" "$BIN_DIR" "$C_RESET" >/dev/tty
      read -r -t 60 _ans </dev/tty || _ans=''
      case "$_ans" in [yY]*) ;; *) die "declined — install uv from https://docs.astral.sh/uv/ and re-run, or pass --yes." ;; esac
    else
      die "uv not found and no terminal to confirm — pass --yes, or install uv from https://docs.astral.sh/uv/ first."
    fi
  fi
  command -v curl >/dev/null 2>&1 || die "need curl to bootstrap uv."
  say "bootstrapping uv into $BIN_DIR ..."
  mkdir -p "$BIN_DIR"
  UV_UNMANAGED_INSTALL="$BIN_DIR" curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$BIN_DIR/uv"
  [ -x "$UV" ] || die "uv bootstrap failed. Install uv from https://docs.astral.sh/uv/ and re-run."
fi
say "uv $("$UV" --version | awk '{print $2}')"

# --- 2. ROCm sanity check --------------------------------------------------
if command -v rocm-smi >/dev/null 2>&1; then
  say "GPU: $(rocm-smi --showproductname 2>/dev/null | head -1 || echo 'ROCm GPU detected')"
else
  warn "rocm-smi not found — proceeding anyway (ROCm may still work via hipcc)."
fi
if command -v hipcc >/dev/null 2>&1; then
  say "hipcc: $(hipcc --version 2>&1 | head -1)"
else
  warn "hipcc not found. GGUF kernels will fail to JIT-compile on first use."
  warn "Install the ROCm toolkit (apt install rocm-hip-sdk) and re-run, or ensure"
  warn "hipcc is on PATH / ROCM_PATH is set."
fi

# --- 3. Create venv and install freetoken -----------------------------------
say "creating venv at $VENV (python $PY_VERSION) ..."
mkdir -p "$FT_HOME"
"$UV" venv "$VENV" --python "$PY_VERSION" --clear

# --- 3a. PyTorch ROCm (from AMD's repo — includes ROCm SDK + hipcc) -----------
# PyPI's torch[rocm] only ships the base wheel.  For RDNA3 we need the
# device-specific extra (device-gfx1100) which pulls in rocm-sdk-core,
# rocm-sdk-libraries, amd-torch-device-gfx1100, etc.  These provide hipcc
# and the libraries needed to JIT-compile GGUF kernels.
say "installing PyTorch (ROCm) from $ROCM_INDEX ..."
"$UV" pip install --python "$VENV" \
  --default-index "$ROCM_INDEX" \
  "torch[device-gfx1100]>=2.11,<2.12"

# --- 3b. Remaining dependencies (from PyPI — torch excluded) -----------------
# pyproject.toml [tool.uv.sources] pins torch to cu130, so we cannot use
# "uv pip install -e '.[dev]'" — it would re-resolve torch from cu130 and
# pull nvidia-cublas/cudnn.  Instead: install all deps explicitly, then
# --no-deps for the editable link.
say "installing freetoken dependencies (from PyPI) ..."
"$UV" pip install --python "$VENV" \
  --extra-index-url https://pypi.org/simple \
  "apache-tvm-ffi==0.1.13.post3" \
  "einops>=0.8,<1" \
  "fastapi>=0.115,<1" \
  "flashlib==0.3.0" \
  "gguf>=0.19,<1" \
  "huggingface_hub>=1.5,<2" \
  "msgpack>=1.1,<2" \
  "modelscope>=1.37,<2" \
  "numpy>=2.0,<2.5" \
  "openai>=2.0,<3" \
  "partial-json-parser>=0.2,<1" \
  "prompt_toolkit>=3.0,<4" \
  "pydantic>=2.9,<3" \
  "pyzmq>=27,<28" \
  "safetensors>=0.6,<1" \
  "tqdm>=4.66,<5" \
  "transformers>=5.5,<6" \
  "triton>=3.6,<3.8" \
  "uvicorn>=0.30,<1" \
  "pytest>=6.0" \
  "setuptools"

# --- 3c. FreeToken editable install (skip dependency resolution) -------------
say "installing freetoken from source (editable, no [accel] — RDNA uses Triton fallbacks) ..."
"$UV" pip install --python "$VENV" \
  --no-build-isolation \
  --no-deps \
  -e .

FT_BIN="$VENV/bin/ft"
[ -x "$FT_BIN" ] || die "install finished but $FT_BIN is missing."

# --- 4. Wire up for PATH + environment.d ------------------------------------
mkdir -p "$BIN_DIR"
ln -sf "$FT_BIN" "$BIN_DIR/ft"
say "symlinked $BIN_DIR/ft -> $FT_BIN"

mkdir -p "$ENV_DIR"
printf 'FREETOKEN_FT_BIN=%s\n' "$FT_BIN" > "$ENV_DIR/50-freetoken.conf"
say "wrote $ENV_DIR/50-freetoken.conf (FREETOKEN_FT_BIN) — GUI picks it up after next login"

# --- 5. Self-check ----------------------------------------------------------
if "$FT_BIN" --help >/dev/null 2>&1; then
  say "self-check: \`ft --help\` OK"
else
  warn "self-check: \`ft --help\` returned non-zero — inspect with: $FT_BIN --help"
fi

cat <<EOF

${C_GREEN}FreeToken engine installed (ROCm).${C_RESET}

  ft binary        $FT_BIN
  on PATH as       $BIN_DIR/ft   (ensure $BIN_DIR is on PATH)
  PyTorch          ROCm ($ROCM_INDEX)
  kernel-cache     skipped (GGUF kernels JIT-compile via hipcc on first use)

Run in this shell without re-login:
  export FREETOKEN_FT_BIN="$FT_BIN"
  ft serve --model <path> --port 1919

First serve will JIT-compile GGUF kernels (~1 min). Subsequent runs use cached .so files.

EOF
