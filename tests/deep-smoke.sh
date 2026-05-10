#!/bin/bash
# Grimoire deep smoke test — GPU-aware model lifecycle test.
# Loads models across all GPUs, tests LRU eviction, protects pinned models.
# Usage: ./deep-smoke.sh [base_url] [api_key] [admin_key]
set -euo pipefail

BASE="${1:-http://localhost:9001}"
KEY="${2:-${GRIMOIRE_API_KEY:-}}"
ADMIN="${3:-${KEY}}"

failures=0
PASS=0
REQUIRE_MODELS=1  # need gpu_count + this many extra models

pass()  { echo "  PASS  $1"; }
fail()  { echo "  FAIL  $1 — $2"; failures=$((failures + 1)); }
abort() { echo "  ABORT $1 — $2"; failures=$((failures + 1)); echo; echo "Result: $failures failures"; exit 1; }
header() { echo; echo "=== $1 ==="; }
die()   { abort "$1" "$2"; }

# ------------------------------------------------------------------
header "prerequisites"
[ -n "$KEY" ]   || die "missing key" "set GRIMOIRE_API_KEY or pass as \$2"
[ -n "$ADMIN" ] || die "missing admin" "set admin key or pass as \$3"
curl -sf "${BASE}/health" > /dev/null || die "grimoire unreachable" "${BASE}/health failed"
pass "grimoire reachable"

header "auth"
code=$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/v1/models")
[ "$code" = "401" ] && pass "unauthenticated rejected"          || fail "auth" "expected 401 got $code"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models")
[ "$code" = "200" ] && pass "authenticated accepted"             || die   "auth" "authenticated request failed"

# ------------------------------------------------------------------
# Gather topology via a single Python helper script
header "topology"
TOPOLOGY=$(BASE="$BASE" KEY="$KEY" python3 <<'PYEOF'
import json, os, urllib.request

BASE = os.environ["BASE"]
KEY  = os.environ["KEY"]

def api(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

models = api("/v1/models")
status = api("/models")
gpu_count = status["gpu_count"]

data = models["data"]
pinned   = [m for m in data if m.get("pinned_gpu") is not None]
nonpinned = [m for m in data if m.get("pinned_gpu") is None]

queue = [m["id"] for m in pinned] + [m["id"] for m in nonpinned]

print(json.dumps({"gpu_count": gpu_count, "queue": queue}))
PYEOF
)

gpu_count=$(echo "$TOPOLOGY" | python3 -c "import sys,json; print(json.load(sys.stdin)['gpu_count'])")
pass "$gpu_count GPU(s) detected"

# Read queue as array
queue_json=$(echo "$TOPOLOGY" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['queue']))")
queue_len=$(echo "$queue_json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
need=$((gpu_count + REQUIRE_MODELS))
[ "$queue_len" -ge "$need" ] || die "models" "need $need models (got $queue_len)"

# First gpu_count models fill GPUs, the rest are eviction triggers
fill_models=$(echo "$queue_json" | python3 -c "import sys,json; a=json.load(sys.stdin); print(' '.join(a[:$gpu_count]))")
eviction_model=$(echo "$queue_json" | python3 -c "import sys,json; a=json.load(sys.stdin); print(a[$gpu_count])")
pass "fill: $fill_models"
pass "eviction trigger: $eviction_model"

# ------------------------------------------------------------------
# helpers
get_status() {
    curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models" | \
        python3 -c "import sys,json; [print(m.get('status',{}).get('value','')) for m in json.load(sys.stdin)['data'] if m['id']==sys.argv[1]]" "$1" 2>/dev/null || echo "unknown"
}

is_pinned() {
    curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models" | \
        python3 -c "import sys,json; [print('yes' if m.get('pinned_gpu') is not None else 'no') for m in json.load(sys.stdin)['data'] if m['id']==sys.argv[1]]" "$1" 2>/dev/null || echo "no"
}

load_model() {
    local model="$1"
    header "load $model"
    local http=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${ADMIN}" -X POST "${BASE}/switch/${model}")
    [ "$http" = "200" ] || die "switch" "got HTTP $http switching to $model"
    pass "switch accepted"

    local deadline=$(($(date +%s) + 180))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local st=$(get_status "$model")
        [ "$st" = "loaded" ]  && { pass "loaded (GPU settled)"; return 0; }
        [ "$st" = "failed" ] && { die "$model" "status is 'failed'"; }
        sleep 2
    done
    die "$model" "timed out waiting for 'loaded'"
}

chat_verify() {
    local model="$1"
    header "chat with $model"
    local resp=$(curl -sf -H "Authorization: Bearer ${KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"max_tokens\":256,\"stream\":false}" \
        "${BASE}/v1/chat/completions" 2>&1) || { fail "chat" "curl failed"; return; }
    local content=$(echo "$resp" | python3 -c "
import sys,json
r=json.load(sys.stdin)
c=r['choices'][0]['message'].get('content','')
if not c: c=r['choices'][0]['message'].get('reasoning_content','')
print(c)
" 2>/dev/null)
    if [ -n "$content" ]; then
        pass "response: $(echo "$content" | head -c 80)"
    else
        fail "chat" "empty response"
    fi
    local ctx=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('context_window',''))" 2>/dev/null || echo "")
    [ -n "$ctx" ] && pass "context_window=$ctx" || fail "context_window" "missing"
}

check_active() {
    local model="$1" expected="$2"
    local active=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/v1/models" | \
        python3 -c "import sys,json; [print(m['active']) for m in json.load(sys.stdin)['data'] if m['id']==sys.argv[1]]" "$model" 2>/dev/null)
    if [ "$active" = "$expected" ]; then
        pass "$model active=$expected"
    else
        fail "active flag" "$model got $active expected $expected"
    fi
}

stop_model() {
    local model="$1"
    header "stop $model"
    local http=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${ADMIN}" -X POST "${BASE}/stop/${model}")
    [ "$http" = "200" ] && pass "stopped" || fail "stop" "got HTTP $http"
    sleep 2
    local st=$(get_status "$model")
    [ "$st" = "unloaded" ] && pass "status is unloaded" || fail "status" "got '$st' expected 'unloaded'"
}

# ------------------------------------------------------------------
header "pre-check"
active_now=$(curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/health" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['active_models']))")
[ "$active_now" = "0" ] && pass "no models pre-loaded" || pass "$active_now models already active"

# ------------------------------------------------------------------
# PHASE 1: Fill all GPUs
header "phase 1: fill all $gpu_count GPU(s)"
loaded_list=""
for model in $fill_models; do
    load_model "$model"
    loaded_list="$loaded_list $model"
    check_active "$model" "True"
    chat_verify "$model"
done
loaded_list=$(echo "$loaded_list" | xargs)
pass "all GPUs filled: $loaded_list"

# ------------------------------------------------------------------
# PHASE 2: Trigger LRU eviction
header "phase 2: LRU eviction"

# Find the oldest non-pinned model (first non-pinned in loaded_list)
oldest=""
for m in $loaded_list; do
    if [ "$(is_pinned "$m")" = "no" ]; then
        oldest="$m"
        break
    fi
done
[ -n "$oldest" ] || die "eviction" "no non-pinned model found to evict"
pass "oldest non-pinned: $oldest"

load_model "$eviction_model"
check_active "$eviction_model" "True"
chat_verify "$eviction_model"

# Check oldest was evicted
old_status=$(get_status "$oldest")
if [ "$old_status" = "unloaded" ]; then
    pass "$oldest LRU-evicted"
else
    fail "eviction" "$oldest still $old_status (expected unloaded)"
fi

# Verify pinned models survived
for m in $loaded_list; do
    [ "$m" = "$oldest" ] && continue
    st=$(get_status "$m")
    [ "$st" = "loaded" ] && pass "$m survived" || fail "eviction" "$m was also evicted"
done

# ------------------------------------------------------------------
# PHASE 3: Cleanup all loaded models
header "phase 3: cleanup"
for m in $loaded_list $eviction_model; do
    st=$(get_status "$m")
    [ "$st" = "unloaded" ] && continue
    stop_model "$m"
done

# ------------------------------------------------------------------
echo
echo "Result: $failures failures"
exit $failures
