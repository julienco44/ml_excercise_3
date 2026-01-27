#!/bin/bash
#
# Run All Experiments for Next Word Prediction
#
# Usage:
#   ./run_all_experiments.sh          # Run all experiments
#   ./run_all_experiments.sh ngram    # Run only N-gram experiments
#   ./run_all_experiments.sh neural   # Run only neural network experiments
#   ./run_all_experiments.sh eval     # Run only evaluation
#

set -e  # Exit on error

# Configuration
DATA_DIR="processed_data"
OUTPUT_DIR="experiments/results"
CHECKPOINT_DIR="experiments/checkpoints"
CONFIG_DIR="experiments/configs"
LATEX_DIR="experiments/results/latex_tables"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LATEX_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run N-gram experiments
run_ngram_experiments() {
    log_info "Running N-gram experiments..."

    echo ""
    echo "=========================================="
    echo "EXPERIMENT 1: N-gram Models"
    echo "=========================================="

    # Bigram models
    log_info "Training 2-gram with Laplace smoothing..."
    python train.py --config "$CONFIG_DIR/bigram_laplace.yaml" || log_warn "Bigram Laplace failed"

    log_info "Training 2-gram with Kneser-Ney smoothing..."
    python train.py --config "$CONFIG_DIR/bigram_kneser_ney.yaml" || log_warn "Bigram KN failed"

    # Trigram models
    log_info "Training 3-gram with Laplace smoothing..."
    python train.py --config "$CONFIG_DIR/trigram_laplace.yaml" || log_warn "Trigram Laplace failed"

    log_info "Training 3-gram with Kneser-Ney smoothing..."
    python train.py --config "$CONFIG_DIR/trigram_kneser_ney.yaml" || log_warn "Trigram KN failed"

    log_info "Training 3-gram with Interpolation smoothing..."
    python train.py --config "$CONFIG_DIR/trigram_interpolation.yaml" || log_warn "Trigram Interp failed"

    # 4-gram models
    log_info "Training 4-gram with Kneser-Ney smoothing..."
    python train.py --config "$CONFIG_DIR/4gram_kneser_ney.yaml" || log_warn "4gram KN failed"

    log_info "N-gram experiments complete!"
}

# Function to run neural network experiments
run_neural_experiments() {
    log_info "Running neural network experiments..."

    echo ""
    echo "=========================================="
    echo "EXPERIMENT 2: Feedforward Neural Network"
    echo "=========================================="

    log_info "Training FNN baseline..."
    python train.py --config "$CONFIG_DIR/fnn_baseline.yaml" || log_warn "FNN baseline failed"

    echo ""
    echo "=========================================="
    echo "EXPERIMENT 3: Recurrent Models"
    echo "=========================================="

    # RNN
    log_info "Training vanilla RNN..."
    python train.py --config "$CONFIG_DIR/rnn_baseline.yaml" || log_warn "RNN baseline failed"

    # LSTM
    log_info "Training LSTM (2 layers)..."
    python train.py --config "$CONFIG_DIR/lstm_baseline.yaml" || log_warn "LSTM baseline failed"

    # GRU
    log_info "Training GRU (2 layers)..."
    python train.py --config "$CONFIG_DIR/gru_baseline.yaml" || log_warn "GRU baseline failed"

    log_info "Neural network experiments complete!"
}

# Function to run cross-validation
run_cross_validation() {
    log_info "Running cross-validation experiments..."

    echo ""
    echo "=========================================="
    echo "EXPERIMENT 4: Cross-Validation"
    echo "=========================================="

    # LSTM cross-validation
    log_info "Running 5-fold CV for LSTM..."
    python cross_validate.py --config "$CONFIG_DIR/lstm_baseline.yaml" \
        --k-folds 5 --output "$OUTPUT_DIR" || log_warn "LSTM CV failed"

    # GRU cross-validation
    log_info "Running 5-fold CV for GRU..."
    python cross_validate.py --config "$CONFIG_DIR/gru_baseline.yaml" \
        --k-folds 5 --output "$OUTPUT_DIR" || log_warn "GRU CV failed"

    log_info "Cross-validation complete!"
}

# Function to run evaluation
run_evaluation() {
    log_info "Running evaluation on all trained models..."

    echo ""
    echo "=========================================="
    echo "MODEL EVALUATION"
    echo "=========================================="

    # Find all checkpoint files
    CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "*.pt" -o -name "*.pkl" 2>/dev/null)

    if [ -z "$CHECKPOINTS" ]; then
        log_warn "No checkpoint files found in $CHECKPOINT_DIR"
        return
    fi

    log_info "Found checkpoints:"
    echo "$CHECKPOINTS"

    # Run evaluation
    python evaluate.py --models $CHECKPOINTS \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT_DIR" \
        --num-samples 15 || log_warn "Evaluation failed"

    log_info "Evaluation complete!"
}

# Function to generate report tables
generate_tables() {
    log_info "Generating LaTeX tables for report..."

    echo ""
    echo "=========================================="
    echo "GENERATING LATEX TABLES"
    echo "=========================================="

    python generate_report_tables.py \
        --results-dir "$OUTPUT_DIR" \
        --ngram-dir "ngram_results" \
        --output-dir "$LATEX_DIR" || log_warn "Table generation failed"

    log_info "LaTeX tables generated in $LATEX_DIR"
}

# Function to run qualitative evaluation
run_qualitative_eval() {
    log_info "Generating samples for qualitative evaluation..."

    echo ""
    echo "=========================================="
    echo "QUALITATIVE EVALUATION"
    echo "=========================================="

    # Find best model (prefer LSTM if available)
    BEST_MODEL=""
    if [ -f "$CHECKPOINT_DIR/lstm_baseline.pt" ]; then
        BEST_MODEL="$CHECKPOINT_DIR/lstm_baseline.pt"
    elif [ -f "$CHECKPOINT_DIR/gru_baseline.pt" ]; then
        BEST_MODEL="$CHECKPOINT_DIR/gru_baseline.pt"
    else
        # Find any neural model
        BEST_MODEL=$(find "$CHECKPOINT_DIR" -name "*.pt" | head -1)
    fi

    if [ -n "$BEST_MODEL" ]; then
        log_info "Using model: $BEST_MODEL"
        python -m evaluation.qualitative_eval generate \
            --model "$BEST_MODEL" \
            --data-dir "$DATA_DIR" \
            --output "$OUTPUT_DIR/samples_for_rating.csv" \
            --num-samples 100 \
            --stratify || log_warn "Qualitative eval failed"

        log_info "Samples saved to $OUTPUT_DIR/samples_for_rating.csv"
    else
        log_warn "No trained models found for qualitative evaluation"
    fi
}

# Main script
main() {
    echo ""
    echo "=========================================="
    echo "NEXT WORD PREDICTION EXPERIMENTS"
    echo "=========================================="
    echo ""

    # Check for argument
    MODE=${1:-"all"}

    case $MODE in
        "ngram")
            run_ngram_experiments
            ;;
        "neural")
            run_neural_experiments
            ;;
        "cv")
            run_cross_validation
            ;;
        "eval")
            run_evaluation
            ;;
        "tables")
            generate_tables
            ;;
        "qualitative")
            run_qualitative_eval
            ;;
        "all")
            run_ngram_experiments
            run_neural_experiments
            run_cross_validation
            run_evaluation
            generate_tables
            run_qualitative_eval
            ;;
        *)
            echo "Usage: $0 [ngram|neural|cv|eval|tables|qualitative|all]"
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "ALL EXPERIMENTS COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo "Checkpoints saved to: $CHECKPOINT_DIR"
    echo "LaTeX tables saved to: $LATEX_DIR"
    echo ""
}

# Run main function
main "$@"
