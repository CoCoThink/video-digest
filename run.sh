#!/usr/bin/env bash
# 在线视频生成字幕系统 — 启动脚本
#
# 用法：
#   ./run.sh                     # 前台启动（默认 :8200）
#   VD_PORT=8300 ./run.sh        # 改端口（任意 VD_* 环境变量均可覆盖，见 server/README.md）
#   VD_PYTHON=/path/to/python ./run.sh   # 指定解释器
#   ./run.sh --bg                # 后台启动，日志写入 ./server.log
#
# 脚本职责：定位 Qwen3-ASR 环境的 Python、预检端口可绑定（规避 WSL2 保留端口段，
# 见《实现与踩坑》§4.5）、然后 `python -m server.main`。启动/关闭钩子由应用自身完成。
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"   # 切到项目根（脚本所在目录）

# ---- 1) 解析 Python 解释器 ------------------------------------------------
pick_python() {
  if [[ -n "${VD_PYTHON:-}" ]]; then echo "$VD_PYTHON"; return; fi
  # 已激活且装好依赖的环境优先
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] \
     && "$CONDA_PREFIX/bin/python" -c 'import qwen_asr' >/dev/null 2>&1; then
    echo "$CONDA_PREFIX/bin/python"; return
  fi
  # 本机 mamba env qwen-asr 的常见位置
  for p in "$HOME/mamba/envs/qwen-asr/bin/python" \
           "$HOME/miniconda3/envs/qwen-asr/bin/python" \
           "$HOME/micromamba/envs/qwen-asr/bin/python"; do
    [[ -x "$p" ]] && { echo "$p"; return; }
  done
  command -v python3 || command -v python   # 兜底：PATH 中的 python
}
PY="$(pick_python)" || true
if [[ -z "${PY:-}" || ! -x "$(command -v "$PY" 2>/dev/null || echo "$PY")" ]]; then
  echo "✗ 未找到可用的 Python。请设置 VD_PYTHON 指向已装 qwen-asr 栈的解释器。" >&2
  exit 1
fi

# ---- 2) 读取监听地址/端口（供预检；应用侧由 pydantic 从环境/.env 再读一次）----
# 从 .env 取某个 VD_ 变量（去掉可能的成对引号）；无 .env 或未定义则空。
get_env() {
  [[ -f .env ]] || return 0
  sed -n "s/^$1=//p" .env | tail -1 | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/"
}
HOST="${VD_HOST:-$(get_env VD_HOST)}"; HOST="${HOST:-0.0.0.0}"
PORT="${VD_PORT:-$(get_env VD_PORT)}"; PORT="${PORT:-8200}"

# ---- 3) 端口预检（WSL2 保留端口段会 bind 失败但 ss 看不到监听者，§4.5）------
if ! "$PY" - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((host, port)); sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
then
  echo "✗ 端口 $HOST:$PORT 无法绑定。" >&2
  echo "  若确无进程监听（ss -ltnp 看不到），多半是 WSL2/Hyper-V 保留端口段（见《实现与踩坑》§4.5）。" >&2
  echo "  换端口重试：  VD_PORT=8300 $0" >&2
  echo "  或在 Windows 管理员执行：  net stop winnat && net start winnat" >&2
  exit 1
fi

# ---- 4) 启动 --------------------------------------------------------------
echo "▶ Python : $PY"
echo "▶ 页面   : http://localhost:$PORT/   （模型在 Worker 子进程加载约 14s，日志「模型就绪」后可上传）"
echo

if [[ "${1:-}" == "--bg" ]]; then
  LOG="${VD_LOG:-./server.log}"
  nohup "$PY" -m server.main >"$LOG" 2>&1 &
  echo "▶ 已后台启动 (pid $!)，日志：$LOG"
  echo "  停止： kill $!    跟踪日志： tail -f $LOG"
else
  exec "$PY" -m server.main
fi
