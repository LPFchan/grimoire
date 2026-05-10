#!/bin/bash
# Grimoire deep smoke test — loads a model, chats with it, switches, unloads.
# Requires a running grimoire instance. Will auto-load models via /switch.
# Usage: ./deep-smoke.sh [base_url] [api_key] [admin_key]
set -euo pipefail

BASE="${1:-http://localhost:9001}"
KEY="${2:-${GRIMOIRE_API_KEY:-}}"
ADMIN="${3:-${KEY}}"

failures=0
active_model=""

pass()  { echo "  PASS  $1"; }
fail()  { echo "  FAIL  $1 — $2"; failures=$((failures + 1)); }
abort() { echo "  ABORT $1 — $2"; failures=$((failures + 1)); echo; echo "Result: $failures failures"; exit 1; }
header() { echo; echo "=== $1 ==="; }

# ------------------------------------------------------------------
die() { abort "$1" "$2"; }

require() {
    if [ -z "${!1:-}" ]; then die "missing $1" "set \$$1"; fi
}

# ------------------------------------------------------------------
header "prerequisites"
require KEY
require ADMIN
if ! curl -sf "${BASE}/health" > /dev/null; then
    die "grimoire unreachable" "${BASE}/health failed"
fi
pass "grimoire reachable"

# ------------------------------------------------------------------
header "auth"
code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/v1/models")
[ "$code" = "401" ] && pass "unauthenticated rejected"    || fail "auth" "expected 401 got $code"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models")
[ "$code" = "200" ] && pass "authenticated accepted"       || fail "auth" "expected 200 got $code"

# ------------------------------------------------------------------
header "registry"
models_json=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models")
count=$(echo "$models_json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
[ "$count" -gt 0 ] && pass "$count models in registry" || die "registry" "empty"

# pick two models from different families for switching test
FIRST=$(echo "$models_json" | python3 -c "
import sys,json
data=json.load(sys.stdin)['data']
qwen=[m['id'] for m in data if m.get('family')=='qwen']
gemma=[m['id'] for m in data if m.get('family')=='gemma']
print(qwen[0] if qwen else data[0]['id'])
")
SECOND=$(echo "$models_json" | python3 -c "
import sys,json
data=json.load(sys.stdin)['data']
gemma=[m['id'] for m in data if m.get('family')=='gemma']
qwen=[m['id'] for m in data if m.get('family')=='qwen']
print(gemma[0] if gemma else data[-1]['id'])
")
pass "first=$FIRST second=$SECOND"

# ------------------------------------------------------------------
load_model() {
    local model="$1"
    header "load $model"
    local http=$(curl -s -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer ${ADMIN}" \
        -X POST "${BASE}/switch/${model}")
    if [ "$http" = "200" ]; then
        pass "switch accepted (200)"
    else
        die "switch" "got HTTP $http switching to $model"
    fi

    # poll until loaded
    local deadline=$(($(date +%s) + 180))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local status=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models" | \
            python3 -c "import sys,json; [print(m.get('status',{}).get('value','')) for m in json.load(sys.stdin)['data'] if m['id']=='$model']" 2>/dev/null || echo "")
        if [ "$status" = "loaded" ]; then
            pass "model loaded"
            active_model="$model"
            return 0
        elif [ "$status" = "failed" ]; then
            die "model $model" "status is 'failed'"
        fi
        sleep 2
    done
    die "model $model" "timed out waiting for 'loaded' status"
}

chat_once() {
    local model="${1:-$active_model}"
    header "chat with $model"
    local resp=$(curl -sf -H "Authorization: Bearer ${KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"max_tokens\":256,\"stream\":false}" \
        "${BASE}/v1/chat/completions" 2>&1)
    local code=$?
    if [ "$code" != "0" ]; then
        fail "chat" "curl failed: $resp"
        return
    fi
    local content=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    if [ -n "$content" ]; then
        pass "response: $(echo "$content" | head -c 80)"
    else
        fail "chat" "empty response"
    fi

    # verify context_window in response
    local ctx=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('context_window',''))" 2>/dev/null || echo "")
    if [ -n "$ctx" ]; then
        pass "context_window=$ctx"
    else
        fail "context_window" "missing in non-streaming response"
    fi
}

check_active_in_models() {
    local model="$1"
    local active=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models" | \
        python3 -c "import sys,json; [print(m['active']) for m in json.load(sys.stdin)['data'] if m['id']=='$model']" 2>/dev/null)
    if [ "$active" = "True" ]; then
        pass "$model marked active in /v1/models"
    else
        fail "active flag" "$model not active in /v1/models"
    fi
}

stop_model() {
    local model="$1"
    header "stop $model"
    local http=$(curl -s -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer ${ADMIN}" \
        -X POST "${BASE}/stop/${model}")
    [ "$http" = "200" ] && pass "stopped" || fail "stop" "got HTTP $http"
    sleep 2
    local status=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models" | \
        python3 -c "import sys,json; [print(m.get('status',{}).get('value','')) for m in json.load(sys.stdin)['data'] if m['id']=='$model']" 2>/dev/null || echo "")
    [ "$status" = "unloaded" ] && pass "status is unloaded" || fail "status" "got '$status' expected 'unloaded'"
    active_model=""
}

# ------------------------------------------------------------------
# Verify none loaded initially
header "pre-check"
active=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/health" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['active_models']))")
[ "$active" = "0" ] && pass "no models loaded" || pass "$active models already loaded (pre-existing)"

# ------------------------------------------------------------------
load_model "$FIRST"
check_active_in_models "$FIRST"
chat_once "$FIRST"

# ------------------------------------------------------------------
load_model "$SECOND"
check_active_in_models "$SECOND"
chat_once "$SECOND"
# first should be evicted (only one GPU pinned model, second takes the other)
old_status=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models" | \
    python3 -c "import sys,json; [print(m.get('status',{}).get('value','')) for m in json.load(sys.stdin)['data'] if m['id']=='$FIRST']" 2>/dev/null || echo "")
if [ "$old_status" = "unloaded" ]; then
    pass "$FIRST auto-evicted on switch"
else
    pass "$FIRST still $old_status (may be on other GPU)"
fi

# ------------------------------------------------------------------
stop_model "$SECOND"

# ------------------------------------------------------------------
echo
echo "Result: $failures failures"
exit $failures
