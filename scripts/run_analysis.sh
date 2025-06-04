#!/bin/bash

# Main runner script for face analysis
# Usage: bash scripts/run_analysis.sh [command] [args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    echo "Face Analysis Tools"
    echo "==================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  single <image>           - Analyze single image"
    echo "  folder <folder>          - Analyze all images in folder"
    echo "  best <folder>            - Select best image from folder"
    echo ""
    echo "Examples:"
    echo "bash $0 single input/test_1.png"
    echo "bash $0 folder input/"
    echo "bash $0 best input/"
    echo ""
    echo "Output will be saved to output/<command>/"
}

case "$1" in
    "single")
        shift
        bash "$SCRIPT_DIR/image_analysis_script.sh" "$@"
        ;;
    "folder")
        shift
        bash "$SCRIPT_DIR/folder_analysis_script.sh" "$@"
        ;;
    "best")
        shift
        bash "$SCRIPT_DIR/best_selection_script.sh" "$@"
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac