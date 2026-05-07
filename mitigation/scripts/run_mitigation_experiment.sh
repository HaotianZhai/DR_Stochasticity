#!/bin/bash
set -e

###############################################################################
# Single Mitigation Experiment Runner
#
# Runs ONE experiment: inference + stochasticity eval + metrics extraction.
# Designed to be called repeatedly by an agent that inspects results between
# runs and decides what to try next.
#
# Usage:
#   export LLM_API_KEY="your-key"
#   export LLM_BASE_URL="https://your-provider/v1"
#   ./run_mitigation_experiment.sh <experiment_name> [OPTIONS]
#
# Options:
#   --temp <float>               Global temperature (default: 0.7)
#   --temp-summarization <float> Summarization module temp
#   --temp-reasoning <float>     Reasoning module temp
#   --temp-query <float>         Query module temp
#   --use-ensemble               Enable query ensemble
#   --use-structure              Enable structured output
#   --use-consistency            Enable consistency voting
#   --rollouts <int>             Number of rollouts (default: 5)
#   --seeds <str>                Comma-separated seed list
#   --max-workers <int>          Parallel workers (default: 10)
#   --dataset <path>             Dataset file (default: ../../data/deepsearchqa.json)
#   --model <str>                Model name
#   --skip-inference             Skip inference, only run eval on existing outputs
#   --skip-eval                  Skip evaluation, only run inference
#   --dry-run                    Print commands without executing
#
# Output:
#   Writes metrics to ablation_outputs/iterative_mitigation/_results/<name>.json
#   Prints a one-line JSON summary to stdout on completion.
#
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================================
# DEFAULTS
# ============================================================================

MODEL="${LLM_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-tput}"
DATASET_FILE="${DATASET_FILE:-${SCRIPT_DIR}/../../data/deepsearchqa.json}"
OUTPUT_BASE="${SCRIPT_DIR}/ablation_outputs/iterative_mitigation"
RESULTS_DIR="${OUTPUT_BASE}/_results"

ROLL_OUT_COUNT=5
MAX_WORKERS=10
SEED_LIST="1,24,32,444,50239"
TEMP="0.7"
TEMP_SUMM=""
TEMP_REAS=""
TEMP_QUERY=""
MITIGATION_FLAGS=""
SKIP_INFERENCE=false
SKIP_EVAL=false
DRY_RUN=false

if [ -z "$LLM_BASE_URL" ]; then
    echo "Warning: LLM_BASE_URL not set. Defaulting to http://localhost:30000/v1" >&2
    export LLM_BASE_URL="http://localhost:30000/v1"
fi
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export LLM_MODEL="${LLM_MODEL:-${MODEL}}"

# ============================================================================
# PARSE ARGS
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [OPTIONS]"
    echo "Run with --help for full options."
    exit 1
fi

EXP_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --temp)                TEMP="$2"; shift 2 ;;
        --temp-summarization)  TEMP_SUMM="$2"; shift 2 ;;
        --temp-reasoning)      TEMP_REAS="$2"; shift 2 ;;
        --temp-query)          TEMP_QUERY="$2"; shift 2 ;;
        --use-ensemble)        MITIGATION_FLAGS="${MITIGATION_FLAGS} --use-ensemble"; shift ;;
        --use-structure)       MITIGATION_FLAGS="${MITIGATION_FLAGS} --use-structure"; shift ;;
        --use-consistency)     MITIGATION_FLAGS="${MITIGATION_FLAGS} --use-consistency"; shift ;;
        --rollouts)            ROLL_OUT_COUNT="$2"; shift 2 ;;
        --seeds)               SEED_LIST="$2"; shift 2 ;;
        --max-workers)         MAX_WORKERS="$2"; shift 2 ;;
        --dataset)             DATASET_FILE="$2"; shift 2 ;;
        --model)               MODEL="$2"; export LLM_MODEL="$2"; shift 2 ;;
        --skip-inference)      SKIP_INFERENCE=true; shift ;;
        --skip-eval)           SKIP_EVAL=true; shift ;;
        --dry-run)             DRY_RUN=true; shift ;;
        --help|-h)
            head -35 "$0" | tail -30
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Fill per-module temps from global if not set
TEMP_SUMM="${TEMP_SUMM:-$TEMP}"
TEMP_REAS="${TEMP_REAS:-$TEMP}"
TEMP_QUERY="${TEMP_QUERY:-$TEMP}"

REFERENCE_FILE="${DATASET_FILE}"
EXP_DIR="${OUTPUT_BASE}/${EXP_NAME}"
MODEL_DIR_NAME="$(basename "${MODEL}")"
DATASET_NAME="$(basename "${DATASET_FILE}")"
ACTUAL_DIR="${EXP_DIR}/${MODEL_DIR_NAME}_modular/${DATASET_NAME}"

# ============================================================================
# VALIDATION
# ============================================================================

if [ -z "$LLM_API_KEY" ] || [ "$LLM_API_KEY" = "EMPTY" ]; then
    echo "Error: LLM_API_KEY not set. Please export LLM_API_KEY with your API key." >&2
    exit 1
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: Dataset not found: ${DATASET_FILE}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_BASE}" "${RESULTS_DIR}"

# ============================================================================
# RUN
# ============================================================================

echo ""
echo "================================================================"
echo " Experiment: ${EXP_NAME}"
echo " Model:      ${MODEL}"
echo " Dataset:    ${DATASET_NAME}"
echo " Temps:      summ=${TEMP_SUMM}  reas=${TEMP_REAS}  query=${TEMP_QUERY}"
echo " Mitigations:${MITIGATION_FLAGS:-  (none)}"
echo " Rollouts:   ${ROLL_OUT_COUNT}  Seeds: ${SEED_LIST}"
echo " Workers:    ${MAX_WORKERS}"
echo "================================================================"

# --- INFERENCE ---
if [ "$SKIP_INFERENCE" = false ]; then
    echo ""
    echo "[1/3] Running inference..."
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] python ../inference/run_multi_react_modular_new.py --model ${MODEL} --output ${EXP_DIR} --dataset ${DATASET_FILE} --temperature ${TEMP} --temp-summarization ${TEMP_SUMM} --temp-reasoning ${TEMP_REAS} --temp-query ${TEMP_QUERY} --roll_out_count ${ROLL_OUT_COUNT} --seed ${SEED_LIST} --max_workers ${MAX_WORKERS} ${MITIGATION_FLAGS}"
    else
        python "${SCRIPT_DIR}/../inference/run_multi_react_modular_new.py" \
            --model "${MODEL}" \
            --output "${EXP_DIR}" \
            --dataset "${DATASET_FILE}" \
            --roll_out_count ${ROLL_OUT_COUNT} \
            --max_workers ${MAX_WORKERS} \
            --temperature ${TEMP} \
            --temp-summarization ${TEMP_SUMM} \
            --temp-reasoning ${TEMP_REAS} \
            --temp-query ${TEMP_QUERY} \
            --seed "${SEED_LIST}" \
            ${MITIGATION_FLAGS}
    fi
    echo "[1/3] Inference complete."
else
    echo "[1/3] Skipping inference (--skip-inference)."
fi

# --- EVALUATION ---
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "[2/3] Running stochasticity + accuracy evaluation..."

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] ../../evaluation/scripts/run_full_stochasticity_pipeline.sh --input-dir ${ACTUAL_DIR} --output-dir ${ACTUAL_DIR}/stochasticity_results --accuracy --reference ${REFERENCE_FILE}"
    else
        if [ ! -d "$ACTUAL_DIR" ]; then
            echo "Error: Trajectory dir not found: ${ACTUAL_DIR}" >&2
            exit 1
        fi
        "${SCRIPT_DIR}/../../evaluation/scripts/run_full_stochasticity_pipeline.sh" \
            --input-dir "${ACTUAL_DIR}" \
            --output-dir "${ACTUAL_DIR}/stochasticity_results" \
            --max-workers ${MAX_WORKERS} \
            --accuracy \
            --reference "${REFERENCE_FILE}"
    fi
    echo "[2/3] Evaluation complete."
else
    echo "[2/3] Skipping evaluation (--skip-eval)."
fi

# --- METRICS EXTRACTION ---
echo ""
echo "[3/3] Extracting metrics..."

RESULT_FILE="${RESULTS_DIR}/${EXP_NAME}.json"
METRICS_FILE="${ACTUAL_DIR}/stochasticity_results/metrics/stochasticity_metrics.json"

if [ "$DRY_RUN" = true ]; then
    cat > "${RESULT_FILE}" << DRYJSON
{"experiment": "${EXP_NAME}", "finding_stoch": 0.85, "answer_stoch": 0.70, "citation_stoch": 0.65, "accuracy": 0.12, "dry_run": true}
DRYJSON
elif [ ! -f "$METRICS_FILE" ]; then
    echo "Warning: Metrics file not found: ${METRICS_FILE}" >&2
    cat > "${RESULT_FILE}" << ERRJSON
{"experiment": "${EXP_NAME}", "error": "metrics not found"}
ERRJSON
else
    python3 << PYEOF
import json
with open("${METRICS_FILE}") as f:
    data = json.load(f)
agg = data.get("aggregate_metrics", {})
acc = data.get("accuracy", {})
result = {
    "experiment": "${EXP_NAME}",
    "finding_stoch": round(agg.get("finding_level_pairwise_stochasticity", -1), 4),
    "answer_stoch": round(agg.get("answer_level_pairwise_stochasticity", -1), 4),
    "citation_stoch": round(agg.get("citation_level_stochasticity", -1), 4),
    "finding_volume_tv": round(agg.get("finding_volume_tv", -1), 2),
    "accuracy": round(acc.get("overall_accuracy", -1), 4),
    "total_correct": acc.get("total_correct", -1),
    "total_questions": acc.get("total_questions", -1),
    "config": {
        "temp_summarization": ${TEMP_SUMM},
        "temp_reasoning": ${TEMP_REAS},
        "temp_query": ${TEMP_QUERY},
        "mitigations": "${MITIGATION_FLAGS}".strip(),
        "rollouts": ${ROLL_OUT_COUNT},
        "seeds": "${SEED_LIST}",
    }
}
with open("${RESULT_FILE}", "w") as f:
    json.dump(result, f, indent=2)
PYEOF
fi

echo "[3/3] Metrics saved to ${RESULT_FILE}"
echo ""

# Print the result
cat "${RESULT_FILE}"
echo ""
echo "================================================================"
echo " Done: ${EXP_NAME}"
echo "================================================================"
