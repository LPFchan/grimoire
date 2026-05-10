#!/bin/bash
# Verify all webui patches apply cleanly and produce valid output.
# Usage: ./tests/webui-patch-build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/../patches"

UPSTREAM_URL="https://github.com/TheTom/llama-cpp-turboquant.git"
UPSTREAM_REF="feature-turboquant-kv-cache-b9079-69d8e4b"

failures=0
pass()  { echo "  PASS  $1"; }
fail()  { echo "  FAIL  $1"; failures=$((failures + 1)); }

echo "=== Cloning upstream ==="
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
git clone --depth 1 --branch "$UPSTREAM_REF" --single-branch "$UPSTREAM_URL" "$TMPDIR/repo" 2>&1 | tail -1

echo "=== Applying patches ==="
for patch in "$PATCHES_DIR"/grimoire-webui-*.patch; do
    name=$(basename "$patch")
    if git -C "$TMPDIR/repo" apply --check "$patch" 2>/dev/null; then
        git -C "$TMPDIR/repo" apply "$patch" 2>/dev/null
        pass "$name"
    else
        fail "$name (apply --check failed)"
    fi
done

echo "=== Balanced HTML check ==="
for f in $(find "$TMPDIR/repo/tools/server/webui/src" -name '*.svelte' 2>/dev/null); do
    opens=$(grep -c '<div\b' "$f" 2>/dev/null || echo 0)
    closes=$(grep -c '</div>' "$f" 2>/dev/null || echo 0)
    if [ "$opens" != "$closes" ]; then
        rel=$(echo "$f" | sed "s|$TMPDIR/repo/||")
        fail "$rel ($opens opens / $closes closes)"
    fi
done

echo
if [ "$failures" -eq 0 ]; then
    echo "Result: all checks passed"
else
    echo "Result: $failures failure(s)"
fi
exit $failures
