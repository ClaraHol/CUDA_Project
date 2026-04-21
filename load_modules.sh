#!/usr/bin/env bash

# Source this script to load required modules in your current shell.
# Usage: source load_modules.sh

if ! command -v module >/dev/null 2>&1; then
    echo "Error: module command not found in this shell."
    echo "Log in to gracy or load your modules environment first."
    return 1 2>/dev/null || exit 1
fi

module load nvhpc/25.9

# ffmpeg is optional for rendering/encoding helpers; prefer 8.1 if available.
preferred_ffmpeg="ffmpeg/8.1"

ffmpeg_module="$({
    module -t avail ffmpeg 2>&1 \
        | awk '
            $1 ~ /^ffmpeg\// {
                gsub(/\(.*\)$/, "", $1)
                print $1
            }
        ' \
        | sort -uV
})"

selected_ffmpeg=""
if printf '%s\n' "$ffmpeg_module" | grep -Fxq "$preferred_ffmpeg"; then
    selected_ffmpeg="$preferred_ffmpeg"
elif [[ -n "$ffmpeg_module" ]]; then
    selected_ffmpeg="$(printf '%s\n' "$ffmpeg_module" | tail -n 1)"
fi

if [[ -n "$selected_ffmpeg" ]]; then
    module load "$selected_ffmpeg"
    echo "Loaded optional module: $selected_ffmpeg"
else
    echo "Warning: no optional ffmpeg module found; continuing without it."
fi

echo "Loaded modules:"
module list 2>&1 | sed -n '/Currently Loaded Modulefiles:/,$p'
