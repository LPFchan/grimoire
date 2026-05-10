#!/bin/bash
# Grimoire smoke test — run against a deployed instance.
# Usage: ./smoke-test.sh [base_url] [api_key]
# Defaults to http://localhost:9001 with GRIMOIRE_API_KEY from env or docker-compose.
set -euo pipefail

BASE="${1:-http://localhost:9001}"
KEY="${2:-${GRIMOIRE_API_KEY:-}}"

failures=0

pass()  { echo "  PASS  $1"; }
fail()  { echo "  FAIL  $1 — $2"; failures=$((failures + 1)); }

header() { echo; echo "=== $1 ==="; }

# ------------------------------------------------------------------
header "health"
if curl -sf "${BASE}/health" > /dev/null; then
    pass "/health"
else
    fail "/health" "no response"
fi

# ------------------------------------------------------------------
header "auth — reject unauthenticated"
code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/v1/models")
if [ "$code" = "401" ]; then
    pass "/v1/models without key returns 401"
else
    fail "/v1/models without key" "got $code, expected 401"
fi

if [ -z "$KEY" ]; then
    echo "  SKIP  no API key — pass GRIMOIRE_API_KEY or supply as \$2"
    echo
    echo "Result: $failures failures"
    exit $failures
fi

# ------------------------------------------------------------------
header "models"
models=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models")
model_count=$(echo "$models" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)
if [ "$model_count" -gt 0 ]; then
    pass "/v1/models returned $model_count models"
else
    fail "/v1/models" "no models returned"
fi

# each model must have context_window
if echo "$models" | python3 -c "
import sys, json
data = json.load(sys.stdin)['data']
missing = [m['id'] for m in data if 'context_window' not in m]
if missing:
    sys.exit(1)
" 2>/dev/null; then
    pass "all models have context_window"
else
    fail "context_window" "some models missing context_window field"
fi

# ------------------------------------------------------------------
header "web UI"
if curl -sf "${BASE}/" 2>/dev/null | grep -qi '<!doctype html>'; then
    pass "/ serves HTML with doctype"
else
    fail "/" "no HTML doctype found"
fi

# ------------------------------------------------------------------
header "chat completions — unknown model"
code=$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${KEY}" \
    -H "Content-Type: application/json" \
    -d '{"model":"nonexistent-model","messages":[{"role":"user","content":"hi"}]}' \
    "${BASE}/v1/chat/completions")
if [ "$code" = "404" ]; then
    pass "/v1/chat/completions with fake model returns 404"
else
    fail "/v1/chat/completions with fake model" "got $code, expected 404"
fi

# ------------------------------------------------------------------
echo
echo "Result: $failures failures"
exit $failures
