#!/usr/bin/env bash

# Source this script to load required modules in your current shell.
# Usage: source load_modules.sh

if ! command -v module >/dev/null 2>&1; then
    echo "Error: module command not found in this shell."
    echo "Log in to gracy or load your modules environment first."
    return 1 2>/dev/null || exit 1
fi

module load nvhpc/25.9

echo "Loaded modules:"
module list 2>&1 | sed -n '/Currently Loaded Modulefiles:/,$p'
