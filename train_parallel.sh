#!/bin/bash

# Parallel training script for sparsity predictors
# Divides layer training across multiple parallel train.py jobs
# 
# Usage:
#   ./train_parallel.sh --layers-per-job 4 --config meta-llama/Llama-2-7b-hf \
#       --dataset_dir ./data/c4 --output_dir ./trained_predictors \
#       --layer_indices 0 1 2 3 4 5 6 7 8 9 10 11 \
#       --batch_size 32 --num_epochs 10 --learning_rate 1e-5
#
# Example with LoRA grid:
#   ./train_parallel.sh --layers-per-job 3 --num_layers 32 --config meta-llama/Llama-2-7b-hf \
#       --dataset_dir ./data/c4 --output_dir ./trained_predictors \
#       --layer_indices all --lora_sizes 4.0 10.0 20.0 30.0 \
#       --batch_size 32 --num_epochs 10 --learning_rate 1e-5

set -e  # Exit on any error

# Default values
LAYERS_PER_JOB=4
TRAIN_ARGS=()
LAYER_INDICES=()
CONFIG=""
NUM_LAYERS=""
PYTHON_CMD="python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 --layers-per-job N [train.py arguments...]

Parallel training script that divides layer training across multiple train.py jobs.

Required arguments:
    --layers-per-job N         Number of layers to train in each parallel job

Script-specific arguments:
    --num_layers N             Total number of layers (required when --layer_indices is 'all')

All other arguments are passed directly to train.py. Key arguments include:
    --config MODEL_PATH        Path to model config (required for train.py)
    --dataset_dir PATH         Path to dataset directory (required for train.py)
    --output_dir PATH          Output directory for trained models (required for train.py)
    --layer_indices LAYERS     Layer indices to train (space-separated numbers or 'all')
    --lora_sizes SIZES         LoRA sizes as percentages (e.g., 4.0 10.0 20.0 30.0)
    --batch_size N             Training batch size
    --num_epochs N             Number of training epochs
    --learning_rate RATE       Learning rate
    --use_wandb                Enable Weights & Biases logging

Examples:
    # Train 12 layers with 4 layers per job (3 parallel jobs)
    $0 --layers-per-job 4 --config meta-llama/Llama-2-7b-hf \\
        --dataset_dir ./data/c4 --output_dir ./trained_predictors \\
        --layer_indices 0 1 2 3 4 5 6 7 8 9 10 11 \\
        --batch_size 32 --num_epochs 10 --learning_rate 1e-5

    # Train all 32 layers with LoRA grid using 3 layers per job
    $0 --layers-per-job 3 --num_layers 32 --config meta-llama/Llama-2-7b-hf \\
        --dataset_dir ./data/c4 --output_dir ./trained_predictors \\
        --layer_indices all --lora_sizes 4.0 10.0 20.0 30.0 \\
        --batch_size 32 --num_epochs 10 --learning_rate 1e-5 --use_wandb

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --layers-per-job)
            LAYERS_PER_JOB="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --layer_indices)
            shift
            # Collect all layer indices until next flag or end
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LAYER_INDICES+=("$1")
                shift
            done
            # Add to train args
            TRAIN_ARGS+=(--layer_indices "${LAYER_INDICES[@]}")
            ;;
        --config)
            CONFIG="$2"
            TRAIN_ARGS+=("$1" "$2")
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            # Pass through all other arguments to train.py
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$LAYERS_PER_JOB" ]] || [[ ! "$LAYERS_PER_JOB" =~ ^[0-9]+$ ]] || [[ "$LAYERS_PER_JOB" -lt 1 ]]; then
    print_error "Invalid --layers-per-job value: '$LAYERS_PER_JOB'. Must be a positive integer."
    show_usage
    exit 1
fi

if [[ -z "$CONFIG" ]]; then
    print_error "Missing required argument: --config"
    show_usage
    exit 1
fi

if [[ ${#LAYER_INDICES[@]} -eq 0 ]]; then
    print_error "Missing required argument: --layer_indices"
    show_usage
    exit 1
fi

print_info "Parallel training configuration:"
print_info "  Layers per job: $LAYERS_PER_JOB"
print_info "  Model config: $CONFIG"
print_info "  Layer indices: ${LAYER_INDICES[*]}"

# Handle 'all' layers by generating the list from num_layers
if [[ ${#LAYER_INDICES[@]} -eq 1 && "${LAYER_INDICES[0]}" == "all" ]]; then
    if [[ -z "$NUM_LAYERS" ]] || [[ ! "$NUM_LAYERS" =~ ^[0-9]+$ ]] || [[ "$NUM_LAYERS" -lt 1 ]]; then
        print_error "When --layer_indices is 'all', you must specify --num_layers with a positive integer"
        print_error "Example: --num_layers 32 --layer_indices all"
        exit 1
    fi
    
    print_info "Generating layer indices for 'all' option with $NUM_LAYERS layers..."
    
    # Generate layer indices from 0 to NUM_LAYERS-1
    LAYER_INDICES=()
    for ((i=0; i<NUM_LAYERS; i++)); do
        LAYER_INDICES+=($i)
    done
    
    print_info "Generated $NUM_LAYERS layers: ${LAYER_INDICES[*]}"
fi

# Calculate number of jobs needed
TOTAL_LAYERS=${#LAYER_INDICES[@]}
NUM_JOBS=$(( (TOTAL_LAYERS + LAYERS_PER_JOB - 1) / LAYERS_PER_JOB ))

print_info "Training plan:"
print_info "  Total layers: $TOTAL_LAYERS"
print_info "  Layers per job: $LAYERS_PER_JOB"
print_info "  Number of parallel jobs: $NUM_JOBS"

# Create output directory for job logs
LOG_DIR="./logs/parallel_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
print_info "Job logs will be saved to: $LOG_DIR"

# Function to run a single training job
run_training_job() {
    local job_id=$1
    local job_layers=("${@:2}")
    local job_name="job_${job_id}"
    local log_file="${LOG_DIR}/${job_name}.log"
    
    print_info "Starting $job_name with layers: ${job_layers[*]}"
    
    # Build the command
    local cmd=("$PYTHON_CMD" "train.py")
    
    # Add all original args except layer_indices
    local skip_next=false
    local in_layer_indices=false
    for arg in "${TRAIN_ARGS[@]}"; do
        if [[ "$skip_next" == true ]]; then
            skip_next=false
            continue
        fi
        
        if [[ "$arg" == "--layer_indices" ]]; then
            in_layer_indices=true
            cmd+=("$arg")
            # Add the job-specific layers
            cmd+=("${job_layers[@]}")
            skip_next=false
        elif [[ "$in_layer_indices" == true && ! "$arg" =~ ^-- ]]; then
            # Skip original layer indices
            continue
        else
            in_layer_indices=false
            cmd+=("$arg")
        fi
    done
    
    print_info "$job_name command: ${cmd[*]}"
    
    # Run the training job
    if "${cmd[@]}" > "$log_file" 2>&1; then
        print_success "$job_name completed successfully"
        return 0
    else
        print_error "$job_name failed! Check log: $log_file"
        return 1
    fi
}

# Start all jobs in parallel
print_info "Starting $NUM_JOBS parallel training jobs..."
JOB_PIDS=()
FAILED_JOBS=()

for ((job_id=0; job_id<NUM_JOBS; job_id++)); do
    start_idx=$((job_id * LAYERS_PER_JOB))
    end_idx=$((start_idx + LAYERS_PER_JOB))
    
    # Get layers for this job
    job_layers=()
    for ((i=start_idx; i<end_idx && i<TOTAL_LAYERS; i++)); do
        job_layers+=("${LAYER_INDICES[i]}")
    done
    
    # Start job in background
    run_training_job "$job_id" "${job_layers[@]}" &
    JOB_PIDS+=($!)
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

print_info "All jobs started. Waiting for completion..."
print_info "Job PIDs: ${JOB_PIDS[*]}"
print_info "You can monitor progress with: tail -f ${LOG_DIR}/*.log"

# Wait for all jobs to complete
FAILED_COUNT=0
for ((i=0; i<${#JOB_PIDS[@]}; i++)); do
    pid=${JOB_PIDS[i]}
    job_id=$i
    
    if wait $pid; then
        print_success "Job $job_id (PID $pid) completed successfully"
    else
        print_error "Job $job_id (PID $pid) failed"
        FAILED_JOBS+=($job_id)
        ((FAILED_COUNT++))
    fi
done

# Summary
echo
echo "==================== TRAINING SUMMARY ===================="
print_info "Total jobs: $NUM_JOBS"
print_info "Successful jobs: $((NUM_JOBS - FAILED_COUNT))"
print_info "Failed jobs: $FAILED_COUNT"

if [[ $FAILED_COUNT -eq 0 ]]; then
    print_success "All training jobs completed successfully!"
    print_info "Check your output directory for trained models"
else
    print_error "Some training jobs failed: ${FAILED_JOBS[*]}"
    print_info "Check the following log files for details:"
    for job_id in "${FAILED_JOBS[@]}"; do
        echo "  ${LOG_DIR}/job_${job_id}.log"
    done
    exit 1
fi

print_info "Log directory: $LOG_DIR"
echo "============================================================" 