#!/bin/bash
# nohup bash gaia1/auto_eval_loop.sh <模型>
set -euo pipefail

usage() {
    cat <<'EOF'
用法: auto_eval_loop.sh <模型名称> [--max-loops N] [--sleep SECONDS] [--num-workers N] [--dataset-path FILE] [--level LEVEL]

后台执行示例:
nohup bash auto_deepsearch.sh eval_qwen3-max-preview-thinking \
  --max-loops 10 \
  --num-workers 5 \
  > output.log 2>&1 &
参数:
  <模型名称>      与 shihan_eval.sh 相同的模型别名，例如 eval_gpt5pro
  --max-loops N   清理后最多重跑的次数 (默认: 5)
  --sleep S       每轮之间的休眠秒数，便于缓冲 (默认: 0)
  --num-workers N 评估使用的工作线程数 (默认: 22)
  --dataset-path  指定评估使用的数据集文件名 (默认: 深度检索评测集.jsonl)，路径固定为 /data1/agent/data/inner_dataset/search/
  --level         指定评估使用的 GAIA 关卡 (默认: 2023_all)
  -h, --help      显示本帮助信息

脚本会循环执行以下步骤，直到达到最大次数或 clean_output.py 没有删除记录:
  1. 同步运行 run_infer.sh 完整评估
  2. 调用 clean_output.py --target <模型名称> 清理错误记录
  3. 如果还有被删除的记录则进入下一轮

所有日志写入 process_logs/gaia/<模型名称>.log 和 process_logs/gaia/<模型名称>_auto_loop.log。
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# cd "${REPO_ROOT}"

MODEL=""
MAX_LOOPS=2
SLEEP_SECONDS=0
NUM_WORKERS=10
DATASET_DIR="/data1/agent/data/inner_dataset/search"
DATASET_FILENAME_DEFAULT="深度检索评测集.jsonl"
DATASET_FILENAME="$DATASET_FILENAME_DEFAULT"
LEVEL_DEFAULT="2023_all"
LEVEL="$LEVEL_DEFAULT"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-loops)
            shift
            [[ $# -gt 0 ]] || { echo "缺少 --max-loops 的值" >&2; usage; exit 1; }
            MAX_LOOPS="$1"
            ;;
        --max-loops=*)
            MAX_LOOPS="${1#*=}"
            ;;
        --sleep)
            shift
            [[ $# -gt 0 ]] || { echo "缺少 --sleep 的值" >&2; usage; exit 1; }
            SLEEP_SECONDS="$1"
            ;;
        --sleep=*)
            SLEEP_SECONDS="${1#*=}"
            ;;
        --num-workers)
            shift
            [[ $# -gt 0 ]] || { echo "缺少 --num-workers 的值" >&2; usage; exit 1; }
            NUM_WORKERS="$1"
            ;;
        --num-workers=*)
            NUM_WORKERS="${1#*=}"
            ;;
        --dataset-path)
            shift
            [[ $# -gt 0 ]] || { echo "缺少 --dataset-path 的值" >&2; usage; exit 1; }
            DATASET_FILENAME="${1##*/}"
            ;;
        --dataset-path=*)
            DATASET_FILENAME="${1#*=}"
            DATASET_FILENAME="${DATASET_FILENAME##*/}"
            ;;
        --level)
            shift
            [[ $# -gt 0 ]] || { echo "缺少 --level 的值" >&2; usage; exit 1; }
            LEVEL="$1"
            ;;
        --level=*)
            LEVEL="${1#*=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -n "$MODEL" ]]; then
                echo "一次只能指定一个模型名称" >&2
                usage
                exit 1
            fi
            MODEL="$1"
            ;;
    esac
    shift || true
done

if [[ -z "$MODEL" ]]; then
    echo "必须提供模型名称" >&2
    usage
    exit 1
fi

if ! [[ "$MAX_LOOPS" =~ ^[0-9]+$ ]] || (( MAX_LOOPS < 1 )); then
    echo "--max-loops 需要是正整数 (当前: $MAX_LOOPS)" >&2
    exit 1
fi

if ! [[ "$SLEEP_SECONDS" =~ ^[0-9]+$ ]] || (( SLEEP_SECONDS < 0 )); then
    echo "--sleep 需要是非负整数 (当前: $SLEEP_SECONDS)" >&2
    exit 1
fi

if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || (( NUM_WORKERS < 1 )); then
    echo "--num-workers 需要是正整数 (当前: $NUM_WORKERS)" >&2
    exit 1
fi

COMMIT_HASH="848f69203366faa405cb535f125f35003c12379e"
AGENT="CodeActAgent"
EVAL_LIMIT=300

RUN_INFER="${SCRIPT_DIR}/evaluation/benchmarks/gaia/scripts/run_infer_interation.sh"
CLEAN_SCRIPT="${SCRIPT_DIR}/clean_output.py"

if [[ ! -f "$RUN_INFER" ]]; then
    echo "未找到评估脚本: $RUN_INFER" >&2
    exit 1
fi

if [[ -z "$DATASET_FILENAME" ]]; then
    echo "数据集文件名不能为空 (--dataset-path)." >&2
    exit 1
fi

if [[ -z "$LEVEL" ]]; then
    echo "关卡参数不能为空 (--level)." >&2
    exit 1
fi

DATASET_PATH="${DATASET_DIR}/${DATASET_FILENAME}"
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "未找到数据集文件: $DATASET_PATH" >&2
    exit 1
fi

if [[ ! -f "$CLEAN_SCRIPT" ]]; then
    echo "未找到清理脚本: $CLEAN_SCRIPT" >&2
    exit 1
fi

LOG_DIR="${SCRIPT_DIR}/process_logs/gaia"
mkdir -p "$LOG_DIR"

EVAL_LOG="${LOG_DIR}/${MODEL}.log"
CONTROL_LOG="${LOG_DIR}/${MODEL}_auto_loop.log"
# touch "$EVAL_LOG" "$CONTROL_LOG"
: >"$EVAL_LOG"          # start a fresh evaluation log each run
touch "$CONTROL_LOG"

log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "$timestamp" "$*" | tee -a "$CONTROL_LOG" "$EVAL_LOG" >/dev/null
}

log "自动评估循环启动: 模型=${MODEL}, 关卡=${LEVEL}, 最大轮数=${MAX_LOOPS}, 间隔=${SLEEP_SECONDS}s, 工作线程=${NUM_WORKERS}"

for (( run = 1; run <= MAX_LOOPS; run++ )); do
    log "开始第 ${run} 轮评估"

    bash "${RUN_INFER}" \
        "${MODEL}" \
        "${COMMIT_HASH}" \
        "${AGENT}" \
        "${EVAL_LIMIT}" \
        "${LEVEL}" \
        "${NUM_WORKERS}" \
        "${DATASET_PATH}" 2>&1 | tee -a "$EVAL_LOG" "$CONTROL_LOG"

    eval_status=${PIPESTATUS[0]}
    if (( eval_status != 0 )); then
        log "评估脚本返回错误状态 ${eval_status}，中止自动循环"
        exit "$eval_status"
    fi

    log "评估完成，开始执行 clean_output.py"

    clean_output=$(python3 "$CLEAN_SCRIPT" --target "$MODEL" --level "$LEVEL" --max_interation 100 2>&1)
    clean_status=$?
    printf '%s\n' "$clean_output" | tee -a "$CONTROL_LOG" "$EVAL_LOG"

    if (( clean_status != 0 )); then
        log "clean_output.py 返回错误状态 ${clean_status}，中止自动循环"
        exit "$clean_status"
    fi

    removed=$(printf '%s\n' "$clean_output" | awk -F': ' '/Removed records:/ {print $2}' | tail -n1 | tr -d '[:space:]')

    if [[ -z "$removed" ]]; then
        log "无法解析 clean_output.py 输出中的删除数量，停止循环"
        exit 1
    fi

    if ! [[ "$removed" =~ ^[0-9]+$ ]]; then
        log "clean_output.py 的删除数量不是数字: ${removed}"
        exit 1
    fi

    if (( removed == 0 )); then
        log "未检测到需要清理的记录，第 ${run} 轮后结束"
        exit 0
    fi

    if (( run == MAX_LOOPS )); then
        log "达到最大轮数 ${MAX_LOOPS}，仍有 ${removed} 条记录被清理，结束循环"
        exit 0
    fi

    if (( SLEEP_SECONDS > 0 )); then
        log "休眠 ${SLEEP_SECONDS} 秒后继续下一轮"
        sleep "$SLEEP_SECONDS"
    fi
done

exit 0
