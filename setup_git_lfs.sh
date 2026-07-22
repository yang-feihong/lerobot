#!/usr/bin/env bash

set -euo pipefail

# Install Git LFS without sudo and initialize hooks for this repository.
#
# Usage in a fresh container:
#   bash setup_git_lfs.sh
#   git push origin main

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

version="3.7.0"
archive_url="https://github.com/git-lfs/git-lfs/releases/download/v${version}/git-lfs-linux-amd64-v${version}.tar.gz"

if command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs already available: $(git-lfs version)"
else
  if [[ -d "$repo_root/.venv/bin" ]]; then
    install_dir="$repo_root/.venv/bin"
  else
    install_dir="$HOME/.local/bin"
  fi

  mkdir -p "$install_dir"

  tmp_dir="/tmp/git-lfs-install-$(date +%Y%m%d_%H%M%S)-$$"
  mkdir -p "$tmp_dir"

  echo "Downloading git-lfs ${version}..."
  curl -L --fail --retry 3 --connect-timeout 20 \
    "$archive_url" \
    -o "$tmp_dir/git-lfs.tar.gz"

  tar -xzf "$tmp_dir/git-lfs.tar.gz" -C "$tmp_dir"
  install -m 755 "$tmp_dir/git-lfs-${version}/git-lfs" "$install_dir/git-lfs"

  export PATH="$install_dir:$PATH"
  echo "Installed git-lfs to: $install_dir/git-lfs"
  echo "git-lfs version: $(git-lfs version)"

  if [[ "$install_dir" == "$HOME/.local/bin" ]]; then
    echo
    echo "Note: add this to your shell startup file if it is not already in PATH:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  fi
fi

git lfs install --local
echo "Git LFS hooks initialized for: $repo_root"
