#!/bin/bash
# Wrapper around opencode that sets OPENCODE_ENABLE_EXA and defaults to qwen-coder.
# Usage (from any project):
#   oc                          # interactive TUI with qwen-coder
#   oc "fix the login bug"      # one-shot
#   oc --agent qwen-chat "hi"   # fast chat mode
#
# Install: symlink into your PATH once:
#   ln -sf /path/to/opencode-mlx-qwen-dflash/scripts/oc.sh /usr/local/bin/oc

export OPENCODE_ENABLE_EXA=true

if [[ $# -eq 0 ]]; then
  exec opencode --agent qwen-coder
elif [[ "$1" == --* ]]; then
  exec opencode "$@"
else
  exec opencode run --agent qwen-coder "$@"
fi
