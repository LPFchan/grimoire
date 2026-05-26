# DEC-20260525-002: Models Page Improvements — Ingest, Files, and Resizable Layout

Opened: 2026-05-26 01:08:02 KST
Recorded by agent: opencode

## Metadata

- Status: proposed
- Deciders: operator
- Related ids: DEC-20260525-001

## Decision

Improve the Models page with three changes: (1) resizable two-panel layout with 35% dynamic left panel, (2) GGUF file management moved into the left panel as a unified scrollable list, (3) HuggingFace URL-based model ingest rendered as an inline placeholder in the left panel with download progress (percentage, speed, ETA) — no modal.

## Context

The v1 Models page (`#/models`, DEC-20260525-001) was deployed with a fixed `w-64` left panel, no visible GGUF file management UI, and no HuggingFace download flow. Three gaps were identified:

1. **Non-resizable divider**: The left panel is hardcoded at 256px.
2. **GGUF file list invisible**: State exists but is never rendered. Delete/rename not wired.
3. **No HuggingFace ingest**: The `/ingest` endpoint downloads synchronously with no progress. No frontend UI.

The existing `/ingest` uses `asyncio.to_thread()` wrapping `urllib.request`, which has no progress callback. The existing `model_filename_from_url()` in `ingest.py` doesn't handle HuggingFace query-param formats.

## Design

### Left Panel Layout

Single scrollable column at 35% of available width (CSS `width: var(--left-panel-pct)`), resizable via drag handle:

```
┌──────────────────────────┬┬──────────────────────────────┐
│ Models              [+] ││                               │
│                          ││                               │
│ Qwen (6)                 ││        Model Editor           │
│  ├ qwen3.6-mtp-27B       ││                               │
│  ├ qwen3.6-mtp-mmproj    ││                               │
│  ├ hereticv2-mtp         ││                               │
│ Gemma4 (4)               ││                               │
│  ├ gemma-4-mtp-31B       ││                               │
│ ──────────────────────── ││                               │
│ GGUF Files (8)       [↑] ││                               │
│  ├ foo.gguf  12MB   ●    ││                               │
│  ├ bar.gguf   5GB   ●    ││                               │
│  ├ orphan.gguf     ⊘  ✕  ││                               │
└──────────────────────────┴┴──────────────────────────────┘
          ↕ drag handle
```

- Default width: 35%, persisted to localStorage. Drag range clamped to [15%, 65%].
- CSS `width` applied via inline style `style="width: var(--left-panel-pct)"` with `min-width: 180px`
- Both model list and GGUF file list in one `overflow-y: auto` container — scroll together
- GGUF section collapsible (header click, state persisted in localStorage)
- Each GGUF row: filename (truncated), size (formatted via `formatFileSize`), status dot (● green = active, ⊘ amber = orphaned)
- Hover reveals: "Rename" (inline edit) and "Delete" (orphaned only, confirmation dialog)
- "Upload" button in GGUF header opens file picker → `POST /registry/upload`

**Resizable divider** — a 4px vertical bar:
- CSS: `cursor-col-resize`, `bg-border/40`, `hover:bg-accent/50`
- Drag via Svelte's native `onpointerdown`/`document.onpointermove`/`onpointerup` — no external library
- Pointer events apply `pointer-events: none` to the rest of the DOM during drag to avoid interference
- Width stored as percentage string (e.g., `"35%"`), stored in `localStorage` key `models-page-left-panel-pct`

### "+" Button and Model Ingest

The "+" button in the Models header opens a small popover (not a modal) with two actions:

```
┌─────────────────────┐
│  Download model     │  → starts HF URL ingest
│  Register local     │  → opens full model editor (existing startNewModel flow)
└─────────────────────┘
```

**"Register local"** — opens the existing full model config editor (`startNewModel()`). No changes to this flow.

**"Download model"** — two-step pop-up with background download:

*Step 1 — URL input*:

```
┌─ Add Model ────────────────────── ✕ ─┐
│ Step 1 of 2                           │
│                                        │
│  HuggingFace URL                       │
│  ┌──────────────────────────────────┐  │
│  │ https://huggingface.co/unslo...  │  │
│  └──────────────────────────────────┘  │
│                                        │
│                               [Next]   │
└────────────────────────────────────────┘
```

Clicking "Next" calls `POST /registry/ingest-start`:
```json
// Request:  { "url": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF?show_file_info=Qwen3.6-27B-Q4_K_M.gguf" }
// Response: { "status": "started", "task_id": "abc123",
//             "filename": "Qwen3.6-27B-Q4_K_M.gguf",
//             "suggested_alias": "qwen3.6-27B-Q4_K_M" }
```

The backend parses the HF URL, resolves the download URL, starts downloading in the background, and returns immediately. If the URL is invalid, returns `{ "status": "failed", "error": "..." }` — pop-up stays at step 1 showing the error.

Immediately after "Next", a **download placeholder row** appears in the left panel BEHIND the still-open pop-up (visible through the pop-up's backdrop/dimmed area):
```
Downloading
 ⟳ Qwen3.6-27B-Q4_K_M.gguf
   ████████░░░░  45%
   7.2 / 16.0 GB
   42 MB/s  ~3m left
```
- Animated spinner + filename (shown from the moment "Next" is clicked)
- Progress bar + percentage
- Downloaded / total size (human-readable)
- Speed and ETA (from server-computed `speed_bytes_per_sec` in the polling response; suppressed on first poll when null)
- If `total_bytes` is null (no Content-Length header), show only downloaded bytes without percentage or ETA: "7.2 GB downloaded"
- Progress continues to update even while the user fills in step 2

If the download completes while the pop-up is still open (step 2), the placeholder shows:
```
Downloading
 ✓ Qwen3.6-27B-Q4_K_M.gguf   (complete)
   Awaiting configuration...
```
The `awaiting_configure` state means bytes hit 100% but `POST /registry/ingest-configure` hasn't been called yet. The spinner stops, replaced by a checkmark.

*Step 2 — Configure* (pop-up transitions immediately after "Next"):

```
┌─ Add Model ────────────────────── ✕ ─┐
│ Step 2 of 2                           │
│                                        │
│  Alias                                 │
│  ┌──────────────────────────────────┐  │
│  │ qwen3.6-27B-Q4_K_M              │  │
│  └──────────────────────────────────┘  │
│                                        │
│  Load settings from                    │
│  ┌──────────────────────────────────┐  │
│  │ (none)                       🔽  │  │
│  └──────────────────────────────────┘  │
│                                        │
│                       [Back]   [Add]   │
└────────────────────────────────────────┘
```

- Alias: pre-filled with `suggested_alias`, editable
- Load settings: dropdown of registered models. Selecting one copies all fields from the source model config except `file`, `mmproj`, `alias`, `added`, `backend`, `mtp-head`, `spec-draft-model`, `draft`, and `drafter` (file-reference fields and backend must not be copied — file fields point to different GGUF files; backend type may differ). The backend then strips incompatible combos:
  - `speculative-type: mtp` + no `mtp-head` → reset speculative-type to none
  - `speculative-type: dflash` + no `spec-draft-model` → reset speculative-type to none
  - `pflash: true` + no `drafter` → reset pflash and park-unpark to none
  - `multimodal`/`vision` capabilities + no `mmproj` → remove those capabilities
  Default "(none)" uses bare defaults (capabilities: completion, ctx-size: 262144).
- **✕** (top-right): closes the pop-up AND cancels the download (calls `DELETE /registry/ingest-status/{task_id}`, removes placeholder, deletes partial file)
- **Back**: cancels the download (same as ✕) and returns to step 1
- **Add**: calls `POST /registry/ingest-configure` → saves config. Pop-up closes.

When the download completes AND "Add" has been clicked, the backend auto-registers the model. The placeholder transitions to a normal model entry in its family group.

If "Add" is clicked and download already completed: registration is immediate, placeholder becomes a model entry instantly.

On download failure: placeholder shows error message with "Retry" and "Remove" buttons. The pop-up (if still open) closes.

### Backend — New Endpoints

All added to `src/grimoire/routes/models.py`:

**`POST /registry/ingest-start`** (parse URL + start download)
```json
// Request:  { "url": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF?show_file_info=Qwen3.6-27B-Q4_K_M.gguf" }
// Success:  { "status": "started", "task_id": "abc123", "filename": "Qwen3.6-27B-Q4_K_M.gguf",
//             "suggested_alias": "qwen3.6-27B-Q4_K_M" }
// Invalid:  { "status": "failed", "error": "URL does not reference a .gguf file" }
```
- Calls `parse_hf_url()` to resolve the download URL + filename from the HF URL
- Starts background download via `asyncio.create_task()`
- `total_bytes` is not returned here — it becomes available in the polling response when the download's Content-Length header is read
- `suggested_alias`: filename minus `.gguf`, lowercased
- Auth: `require_api` (follows DEC-20260525-001 precedent: all registry management endpoints use `require_api`, same trust boundary as plugin toggling. Note: the older `/ingest` endpoint uses `require_admin` but is a legacy design — the new `/registry/*` namespace consistently uses `require_api`)

**`POST /registry/ingest-configure`** (save config for pending download)
```json
// Request:  { "task_id": "abc123", "alias": "qwen3.6-27B-Q4_K_M",
//             "load_settings_from": "qwen3.6-mtp-27B" }
// Response: { "status": "configured" }
```
- Saves `alias` and optional `load_settings_from` into `_ingest_tasks[task_id]`
- Rejects with 400 if task status is `"cancelled"` or `"failed"` (non-configurable terminal states)
- Runs `_validate_model_config()` on the merged config (template defaults + user input) before storing — rejects with 400 if invalid
- If the download is already complete, registers the model immediately (merging settings)
- If the download is still running, stores as pending — registration happens on download completion
- `load_settings_from`: optional model name whose config is copied as template. Validated against `registry.list_all()` — returns 404 if the referenced model doesn't exist. If `null`/omitted (UI "(none)"), uses bare defaults
- If `task_id` is not in `_ingest_tasks` (pruned, cancelled, or never existed), returns 404
- Auth: `require_api`

**`GET /registry/ingest-status/{task_id}`**
```json
// During download
{ "status": "downloading", "filename": "...", "suggested_alias": "...",
  "downloaded_bytes": 7200000000, "total_bytes": 16000000000,
  "speed_bytes_per_sec": 44000000, "eta_seconds": 182 }
// total_bytes: null if Content-Length header is absent
// speed_bytes_per_sec/eta_seconds: null on first poll (no delta yet)
// Done — model registered
{ "status": "done", "filename": "...", "alias": "..." }
// Done — download ready, waiting for configure
{ "status": "awaiting_configure", "filename": "...", "suggested_alias": "..." }
// Failed
{ "status": "failed", "error": "Connection reset" }
```
- `awaiting_configure`: download finished but `POST /registry/ingest-configure` hasn't been called yet (user still on step 2)
- Reads from `_ingest_tasks` dict. Computes speed from last two poll timestamps and byte deltas
- Auth: `require_api`

**`DELETE /registry/ingest-status/{task_id}`** (cancel)
- Checks task status first: if not `"downloading"` or `"awaiting_configure"`, returns `{ "status": "already_finished" }`
- Sets `cancelled = True` flag (checked by download loop between chunks); waits 1 second for the chunk-loop to notice the flag before proceeding
- Removes the `.part` file if it still exists
- Returns `{ "status": "cancelled" }`
- Auth: `require_api`

**`PATCH /registry/gguf`** (rename)
```
PATCH /registry/gguf?filename=gguf/old.gguf&new_filename=gguf/new.gguf
```
- Checks `new_filename` does not already exist on disk (returns 409 if so)
- Renames file on disk (containment: `realpath` on parent dir of `new_filename` since the file doesn't exist yet; `os.path.commonpath` check against `MODELS_DIR`)
- Updates all model configs referencing old filename (scans `file`, `mmproj`, `mtp-head`, `spec-draft-model`, `draft`, `drafter`)
- Saves updated registry
- Returns `{ "status": "renamed", "updated_models": ["model-a"] }`
- Auth: `require_api` (DEC-20260525-001 precedent)

### Backend — New Helpers in `ingest.py`

**`parse_hf_url(raw_url: str)`**:
```python
def parse_hf_url(raw_url):
    """Resolve a HuggingFace URL to a concrete download URL and filename."""
    parsed = urllib.parse.urlparse(raw_url)
    params = urllib.parse.parse_qs(parsed.query)
    segments = [s for s in parsed.path.strip("/").split("/") if s]

    # Case 1: ?show_file_info=foo.gguf OR ?download=foo.gguf — extract filename + branch from path
    # Only treat as filename if the value ends with .gguf
    filename = (params.get("show_file_info") or [None])[0]
    maybe_download = (params.get("download") or [None])[0]
    if not filename and maybe_download and maybe_download.lower().endswith(".gguf"):
        filename = maybe_download
    if filename and filename.lower().endswith(".gguf"):
        if len(segments) >= 2:
            user, repo = segments[0], segments[1]
            # Extract branch from /blob/<branch>/... or /tree/<branch>/...
            branch = "main"
            for i, s in enumerate(segments):
                if s in ("blob", "tree") and i + 1 < len(segments):
                    branch = segments[i + 1]
                    break
            dl_url = f"https://huggingface.co/{user}/{repo}/resolve/{branch}/{filename}"
            return dl_url, filename

    # Case 2: /user/repo/blob/<branch>/<subpath...>/<filename>.gguf or .../resolve/...
    # Check last path component ends with .gguf, not arbitrary substring
    fn_check = parsed.path.lower().rstrip("/")
    if fn_check.endswith(".gguf") or any(seg.lower().endswith(".gguf") for seg in fn_check.split("/")):
        parts = [s for s in parsed.path.strip("/").split("/") if s]
        idx = next((i for i, p in enumerate(parts) if p in ("blob", "tree", "resolve")), None)
        if idx is not None and len(parts) > idx + 2 and parts[-1].lower().endswith(".gguf"):
            user, repo = parts[0], parts[1]
            branch = parts[idx + 1]
            filename = parts[-1]
            subpath = "/".join(parts[idx + 2:-1])
            if subpath:
                dl_url = f"https://huggingface.co/{user}/{repo}/resolve/{branch}/{subpath}/{filename}"
            else:
                dl_url = f"https://huggingface.co/{user}/{repo}/resolve/{branch}/{filename}"
            return dl_url, filename

    # Case 3: Direct URL ending in .gguf — use as-is
    fn = os.path.basename(urllib.parse.unquote(parsed.path))
    if fn.lower().endswith(".gguf"):
        return raw_url, fn

    raise ValueError("URL does not reference a .gguf file")
```

**`download_model_file_with_progress(url, target_path, task_id, progress_dict, max_bytes, timeout)`**:
- Rewrites the chunk loop from `download_model_file()` to inject progress tracking (can't wrap the existing function — it has an internal `while True` loop with no callback mechanism)
- Replicates all existing URL validation: `validate_ingest_url()` before opening, `SafeRedirectHandler` for redirect targets, and re-validation of the final URL after redirects
- After each `CHUNK_SIZE` chunk: updates `progress_dict["downloaded_bytes"]` and reads `total_bytes` from Content-Length header
- Checks `progress_dict["cancelled"]` flag between chunks; raises `CancelledError` if set
- On completion: atomically renames `.part` → final path via `os.replace()`, sets status to "awaiting_configure" or "done" depending on whether config was submitted
- On error: sets status to "failed", cleans up `.part`

**Task cleanup**: A background coroutine `prune_old_ingest_tasks()` runs every 10 minutes:
- Removes `_ingest_tasks` entries whose download completed (or was cancelled/failed) ≥1 hour ago
- Deletes orphaned `.part` files in `MODELS_DIR/gguf/` older than 1 hour (cleans up after server restarts)
- For tasks in `awaiting_configure` or `failed` state where the `.gguf` file exists but is not in the registry, deletes the file (abandoned downloads)
- Started via FastAPI lifespan handler (`@router.on_event("startup")` or equivalent) — not at module level

**_ingest_tasks entry schema** (module-level `dict[str, dict]`):
| Field | Type | Purpose |
|-------|------|---------|
| `status` | `"downloading"\|"done"\|"awaiting_configure"\|"failed"\|"cancelled"` | Current state |
| `filename` | string | GGUF filename |
| `suggested_alias` | string | Auto-derived model name |
| `downloaded_bytes` | int | Bytes downloaded so far |
| `total_bytes` | int\|null | From Content-Length, null if unknown |
| `tmp_path` | string | Path to `.part` file (for cancel/cleanup) |
| `target_path` | string | Final file path |
| `cancelled` | bool | Set to true by cancel endpoint |
| `configured` | bool | Set when `POST /registry/ingest-configure` is called |
| `alias` | string\|null | User-configured alias (from configure) |
| `load_settings_from` | string\|null | Source model for settings copy |
| `error` | string\|null | Error message if failed |
| `last_poll_at` | float | Timestamp of last status poll (for speed calc) |
| `last_poll_bytes` | int | Bytes at last poll (for speed calc) |

Speed: `(downloaded_bytes - last_poll_bytes) / (now - last_poll_at)`. ETA: `(total_bytes - downloaded_bytes) / speed`.
`last_poll_at` and `last_poll_bytes` are initially `None` — speed and ETA are `null` until at least two polls have occurred.

**Concurrency**: All `_ingest_tasks` dict-entry mutations are serialized by an `asyncio.Lock`. The lock is held only during dict read/write operations — not during blocking I/O (chunk reads, file writes) — so polls, configure, and cancel remain responsive.

### Frontend Changes

**Store**: `models-registry.svelte.ts` — add:
```
ingestTask: { task_id: string, filename: string, status: string, downloaded_bytes: number,
              total_bytes: number, speed_bytes_per_sec: number, eta_seconds: number,
              suggested_alias: string, configured: boolean, error: string } | null

startIngest(url: string) → void (POST /registry/ingest-start, sets ingestTask, starts polling)
configureIngest(task_id: string, alias: string, loadSettingsFrom: string | null) → void
    (POST /registry/ingest-configure, sets configured flag; if download already done, transitions to model entry)
cancelIngest() → void (DELETE task, clears ingestTask)
renameGguf(oldName, newName) → void (PATCH /registry/gguf)
```

The store manages the `ingestTask` state and polls `GET /registry/ingest-status/{task_id}` every 1 second. `speed_bytes_per_sec` and `eta_seconds` come from the server polling response (server tracks the last two poll timestamps and byte deltas).

**Page component**: `webui/src/routes/models/+page.svelte` — changes:
- Replace `class="w-64"` on `<aside>` with dynamic width via CSS variable
- Add resize drag handle between panels
- Restructure left panel: model list at top, Separator, GGUF files at bottom (collapsible)
- Add GGUF rows with hover actions (rename inline, delete with confirmation)
- Replace "+" button with popover menu (Download model / Register local). "Download model" is disabled while an active download is in progress (shows "Download in progress..." tooltip instead)
- Add inline download prompt and download placeholder rendering in the left panel
- Progress bar, speed, and ETA computed from store's `ingestTask` state
- Remove unused `showGgufTab` state; remove standalone file upload section in right panel (GGUF management is now only in the left panel — the existing right-panel inline upload in the GGUF file selector dropdown is kept)

### Validation

- `parse_hf_url()` is called inside `POST /registry/ingest-start` — invalid URLs fail immediately with an error response, no download started
- `?show_file_info=` / `?download=` format (Case 1) only resolves to repo root — for files in subdirectories, use the `/blob/<branch>/subdir/file.gguf` URL format (Case 2)
- size limit check: when the download task reads Content-Length, if it exceeds `GRIMOIRE_INGEST_MAX_BYTES`, the task is cancelled and the status switches to "failed"
- containment checks on rename/delete GGUF (existing pattern from DEC-20260525-001; `DELETE /registry/gguf` endpoint is pre-existing)
- `new_filename` collision on rename: returns 409 if the target filename already exists
- alias collision: `registry.add()` raises `ValueError` if model name exists; frontend shows error
- `load_settings_from`: validated against `registry.list_all()` — returns 404 if the referenced model doesn't exist. Passed as `null` (or omitted) when "(none)" is selected in the UI — the endpoint treats `null` as "use defaults" with no model lookup

### Files Changed

| File | Change |
|------|--------|
| `src/grimoire/routes/models.py` | Add `POST /registry/ingest-start`, `POST /registry/ingest-configure`, `GET /registry/ingest-status/{task_id}`, `DELETE /registry/ingest-status/{task_id}`, `PATCH /registry/gguf`; add `_ingest_tasks` dict and `prune_old_ingest_tasks()` |
| `src/grimoire/ingest.py` | Add `parse_hf_url()`, rewrite `download_model_file_with_progress()` (inline chunk loop with progress callbacks) |
| `webui/src/routes/models/+page.svelte` | Resizable divider, unified left panel, ingest UI, GGUF rename/delete wiring, download placeholder rendering |
| `webui/src/lib/stores/models-registry.svelte.ts` | Add ingest state, URL parsing, polling, cancel, rename methods |

### Known Limitations

- **Download resume**: No resume-after-disconnect. Restart required on failure.
- **Server restart**: `_ingest_tasks` is in-memory. On server restart, downloads are lost and `.part` files become orphans. Cleanup on startup could be added later.
- **`awaiting_configure` orphan**: If user navigates away while download is complete but config not submitted, the `.gguf` file exists on disk but is not registered. Prune runs after 1 hour.
- **Model rename**: Still not supported at the model level (only GGUF rename). Delete + re-add is the workaround.
- **Multi-download**: Only one active download per page load. The "Download model" button is disabled while a download is active.
- **Back button data loss**: Clicking "Back" in step 2 cancels the download. The URL must be re-typed.
- **Selected model state duplication**: The store has `selectedModel` and the page has `selectedModelName` — a pre-existing issue not addressed here.

## Consequences

- GGUF files visible alongside models in a unified left panel
- HuggingFace URLs auto-parse to extract filename and construct download URL, respecting branch info
- Download progress shows inline in the model list with speed and ETA
- Divider width persisted between sessions
- No modal — everything happens in the left panel
