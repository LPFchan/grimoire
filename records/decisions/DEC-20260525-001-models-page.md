# DEC-20260525-001: Models Page — Registry and GGUF Manager in WebUI

Opened: 2026-05-25 15:33:01 KST
Recorded by agent: opencode

## Metadata

- Status: proposed
- Deciders: operator
- Related ids: none

## Decision

Add a **"Models" page** (`#/models`) to the grimoire webui sidebar that serves as a visual manager for `models.json` entries and the GGUF file directory. The page supports viewing, editing, adding, and deleting model configurations, as well as uploading and removing orphaned GGUF files. Write endpoints use `require_api` (same pattern as `/stats/plugins` mutations), not a separate admin token.

## Context

The model registry (`/var/lib/grimoire/models.json`) and GGUF directory (`MODELS_DIR/gguf/`) are currently managed only via CLI — there is no web UI for any CRUD operations. The `/ingest` backend endpoint can download models from URLs but has no frontend. Models recently had to be removed manually (CLI edit + `rm gguf`), exposing the gap.

The webui is a SvelteKit fork of `ggerganov/llama.cpp` tools/server/webui (hosted at `github.com:LPFchan/llama-cpp-webui.git`). Custom grimoire pages (`dashboard/`, `plugins/`, `mcp-servers/`) are added as sibling SvelteKit route directories. The dashboard and plugins pages both follow this pattern: a `+page.svelte` component, a `+page.ts` with `validateApiKey(fetch)`, and a sidebar nav icon entry in `SIDEBAR_ACTIONS_ITEMS` (in `constants/ui.ts`). The mcp-servers page is an exception — it has no `+page.ts` — but this plan follows the dashboard/plugins pattern.

The backend `ModelRegistry` class (`src/grimoire/registry.py`) has `add()`, `update()`, `remove()`, `list_all()`, `list_fixed()`, `get_family_defaults()`, and `validate()`. Only `add()` is exposed via the `/ingest` endpoint. The remaining methods have no HTTP endpoints. Additionally, the existing `/ingest` only supports downloading from URLs — it cannot handle direct file uploads.

The `models.json` schema has three top-level keys: `models` (per-model configs), `fixed` (GPU pinning), and `family_defaults` (inheritance — e.g., all qwen family models inherit `preserve_thinking: true`). The `family_defaults` block is currently invisible to webui users.

### Important schema distinction: speculative-type vs pflash

The backend (`model_manager.py`) treats these as orthogonal:
- `speculative-type`: a string field (`"nextn"`, `"mtp"`, `"dflash"`) that controls `--spec-type` on the llama-server CLI
- `pflash`: a **top-level boolean** field (`"pflash": true`) that triggers a separate pflash daemon subprocess (`_start_pflash_daemon()`). It is not a value of `speculative-type`

The editor must reflect this: `speculative-type` is a dropdown; pflash is a separate toggle.

## Options Considered

### Keep editing models.json via CLI

- Upside: zero implementation cost
- Downside: operator-only, requires SSH access, error-prone (JSON syntax, missing fields)

### Extend the existing settings page with a models section

- Upside: reuses existing settings layout
- Downside: settings is already dense; models management is conceptually distinct (file management + registry editing, not preferences); conflating them harms navigation clarity

### Build a dedicated Models page

- Upside: clean separation of concerns, room for growth (file upload, validation feedback, orphan detection), follows existing page pattern
- Downside: new route, new backend endpoints, moderate implementation cost

## Rationale

The CLI-only workflow is operator-gated and error-prone. A visual editor reduces friction for routine tasks and exposes orphaned GGUF files for cleanup. The implementation follows the established dashboard/plugins page pattern and reuses existing infrastructure (`apiFetch`, shadcn-svelte components, `ModelRegistry` CRUD methods). The diff is contained: 4 files changed/created on the frontend, new endpoints added to `models.py` on the backend.

### Backend Design

New endpoints in `src/grimoire/routes/models.py` (existing router; `/registry` namespace separates registry management from model-serving endpoints like `/v1/models` and `/models/load`):

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/registry` | `require_api` | Full registry snapshot + GGUF directory scan |
| `GET` | `/registry/model/{name}` | `require_api` | Single model config |
| `PUT` | `/registry/model/{name}` | `require_api` | Upsert a model entry |
| `DELETE` | `/registry/model/{name}` | `require_api` | Remove entry; `?delete_gguf=true` deletes main GGUF; `?delete_mtp_head=true` deletes MTP head |
| `DELETE` | `/registry/gguf` | `require_api` | Delete orphaned GGUF (`?filename=gguf/foo.gguf`); query param avoids path-parameter slashes |
| `POST` | `/registry/upload` | `require_api` | Multipart upload of a `.gguf` file to `MODELS_DIR/gguf/` |

**Auth**: All endpoints use `require_api`, not `require_admin`. This follows the precedent set by `PATCH /stats/plugins/{key}` (plugin toggle — a mutation endpoint — uses `require_api`). The webui login flow (`auth.py:95-120`) injects `apiKey` into localStorage; there is no separate admin token flow. Requiring `require_admin` would make the page unusable from the webui without additional login infrastructure.

**`GET /registry` response shape**:
```json
{
  "models": { "<name>": { ... } },
  "fixed": { "<name>": <gpu_id> },
  "family_defaults": { "<family>": { ... } },
  "active": ["running-model-1"],
  "gpu_count": 2,
  "gguf_files": [
    { "filename": "gguf/foo.gguf", "size_bytes": 12000000, "used_by": ["model-a", "model-b"] },
    { "filename": "gguf/orphan.gguf", "size_bytes": 5000000, "used_by": null }
  ],
  "mmproj_files": [
    { "filename": "gguf/mmproj.gguf", "size_bytes": 2000000, "used_by": ["model-a", "model-b"] }
  ],
  "gguf_dir": "/models/gguf"
}
```
- `active`: from `manager.list_active()`, enables the frontend to highlight running models
- `gguf_files`: all `.gguf` files in `MODELS_DIR/gguf/` (flat scan). `used_by` is an array of model names referencing the file (`null` if orphaned). A file is considered "referenced" if it appears in any of these model config fields: `file`, `mmproj`, `mtp-head`, `spec-draft-model`, `draft`, `drafter`
- `mmproj_files`: subset of `.gguf` files used as multimodal projectors, listed separately so the editor can offer a dedicated mmproj selector. `used_by` is an array (projectors may be shared). These files also appear in `gguf_files`
- `gguf_dir`: the absolute path to the GGUF directory (for display purposes; frontend should not use this to construct file paths — use `filename` fields directly)

**`PUT /registry/model/{name}` — Upsert logic**:
`ModelRegistry` has no `upsert()` method. The handler implements:
```python
_validate_model_config(body)  # structural checks (types, required fields)
try:
    result = registry.add(name, body)
except ValueError:  # model already exists
    result = registry.update(name, body)
# Post-save validation: file existence, DFlash GGUF contract, GPU range
valid, msg = registry.validate(name, gpu_count=manager.gpu_count)
# Note: validate() returns (False, "Model 'name' not found") for new models
# that haven't been added yet, hence the try/except before validate().
# After add/update, the model is in the registry so validate() works.
return JSONResponse(
    content={"model": result, "valid": valid, "message": msg},
    status_code=201 if created else 200
)
```
`_validate_model_config()` checks required fields and types (equivalent of client-side rules below) before touching the registry.

**`DELETE /registry/gguf?filename=...` — Security**:
1. Resolve: `real_path = os.path.realpath(os.path.join(MODELS_DIR, filename))`
2. Containment check: `os.path.commonpath([os.path.realpath(MODELS_DIR), real_path]) == os.path.realpath(MODELS_DIR)`
3. After containment is confirmed: verify the file is not referenced by any model entry (scanning `file`, `mmproj`, `mtp-head`, `spec-draft-model`, `draft`, `drafter` fields across all models)
4. Delete with `os.remove(real_path)`

Using `os.path.realpath()` before `commonpath()` prevents `../` escapes and symlink traversal.

**`DELETE /registry/model/{name}` — Two-step shared-file gate**:

Step 1: The endpoint always removes the registry entry first (`registry.remove(name)`).

Step 2: If `?delete_gguf=true` or `?delete_mtp_head=true` is set, scan all remaining model entries for references to the file(s). If the file is referenced by other models, **do not delete it** — instead return:
```json
{ "status": "removed", "shared_files": { "gguf/main.gguf": ["other-model-1", "other-model-2"] } }
```
The frontend displays a second confirmation: "This GGUF is still referenced by: other-model-1, other-model-2. Delete anyway?" with a "Force delete" button that re-sends the request with `?force=true`. When `?force=true`, the endpoint skips the reference check and deletes the file (after containment validation via `realpath` → `commonpath`).

The `mmproj` file is never deleted (it is frequently shared). The containment check (`realpath` → `commonpath`) applies before any `os.remove()` call.

**`POST /registry/upload` — Constraints**:
- Max file size: 40 GB (configurable via env var, default matches largest expected GGUF)
- File type: must end in `.gguf` (case-insensitive)
- On disk full or write failure: returns 507 Insufficient Storage or 500 with a descriptive error
- Saves to `MODELS_DIR/gguf/<sanitized_filename>` (basename only, no path components accepted in the upload filename)

The frontend uses raw `fetch` with `FormData` for this endpoint — `apiFetch()` hardcodes `Content-Type: application/json` and cannot be used for multipart requests. A thin `uploadGguf(file: File)` helper in the store handles this.

### Frontend Design

**Files changed/created**:

| File | Action |
|------|--------|
| `webui/src/routes/models/+page.svelte` | Create — main page component |
| `webui/src/routes/models/+page.ts` | Create — API key validation guard |
| `webui/src/lib/stores/models-registry.svelte.ts` | Create — Svelte 5 reactive store for registry data |
| `webui/src/lib/constants/ui.ts` | Modify — add sidebar nav icon entry |

No changes to `webui/src/lib/constants/routes.ts`. Following the dashboard/plugins convention, the route is a raw string `'#/models'`, not a `ROUTES` constant.

**Sidebar nav item** (in `SIDEBAR_ACTIONS_ITEMS`):
```typescript
{ icon: Boxes, tooltip: 'Models', route: '#/models', activeRouteId: '/models' }
```
Import `Boxes` from `@lucide/svelte`. `activeRouteId: '/models'` ensures the icon highlights when on the models page.

**Store boundary** (`models-registry.svelte.ts` vs existing `models.svelte.ts`):
- `models.svelte.ts` (757 lines, existing): manages runtime model state — loaded/loading/unloaded status, model selection, favorites, props cache, modality detection. Used by the chat UI.
- `models-registry.svelte.ts` (new): manages model configuration — the `models.json` entries and GGUF files. Used only by the models management page.

The two stores are independent; the registry store does not feed into the models store.

**Page layout** — two-panel inside root `Sidebar.Inset`:

*Left panel* (model list):
- Grouped by `family` field. Sections for each family with model count. "Ungrouped" section for models without a family.
- Each row: model name, truncated GGUF file path, active indicator (green dot if model is in `active` list)
- Click to select → loads right panel editor
- "+ Add Model" button at top, visually distinct

*Right panel* (editor, shown when a model is selected):

| Group | Fields |
|-------|--------|
| Basic | Model name (read-only for existing, editable for new), GGUF file (dropdown of `gguf_files` + inline upload button), Family (text with autocomplete from known families in `family_defaults`), Backend (dropdown: `llama`) |
| Family defaults | **Read-only hint**: "Family 'qwen' inherits: `extra-args: ["--chat-template-kwargs", "..."]`". Shown when the selected model's `family` has an entry in `family_defaults`. Editing family defaults is deferred to a follow-up DEC |
| Speculative | Speculative type (dropdown: `none`, `nextn`, `mtp`, `dflash`). Conditional fields: `mtp` → MTP head file selector (required). `dflash` → DFlash draft file selector (required; writes to `spec-draft-model`, the canonical field name). The legacy `draft` field is read as a fallback but stripped from config on save (migrated to `spec-draft-model`) |
| PFlash | **Separate toggle** (not part of speculative-type dropdown). When enabled: writes `"pflash": true` to config + shows drafter file selector (required) and park/unpark toggle (`park-unpark` field). The backend (`model_manager.py:483-486`) checks `cfg.get("pflash")` as a boolean, not `speculative-type == "pflash"` |
| Extra args | Ordered flat list of editable strings (add/remove rows). Stored as `["--flag", "value", ...]` — not key-value pairs. Future: tooltip hints per speculative type (e.g., nextn suggests `--draft-max`, `--draft-min`; mtp suggests `--draft-block-size`) |
| Capabilities | Checkboxes: `completion`, `multimodal`, `vision`. Checking `multimodal` auto-checks `vision` in the UI; on save, writes `"multimodal"` only (not both), since the backend already infers vision from multimodal (`_synthetic_props()` in models.py:28). `vision` can be independently checked without multimodal |
| Context | `ctx-size`, `predict`, `parallel` (all positive integers; `parallel` defaults to 1) |
| GPU/Cache | `n-gpu-layers` (default 999), `cache-type-k`, `cache-type-v` (dropdown: `q8_0`, `q4_0`, `turbo4`, `f16`) |
| Multimodal | Conditionally visible when `multimodal` or `vision` capability is set. `mmproj` file (select from `mmproj_files` list with search), `image-min-tokens`, `image-max-tokens` |
| Cost | `input`, `output`, `cache_read` (non-negative numbers) |

**GGUF file selector UX**: The dropdown lists all `.gguf` files from the `gguf_files` array with filename and size. An inline "+" button next to the dropdown opens a file picker for direct upload — avoiding the two-step workflow of navigating to a separate tab first. On successful upload, the dropdown refreshes and the new file is auto-selected.

Actions bar (sticky at bottom of editor):
- **Save** button → calls `PUT /registry/model/{name}`, refreshes registry data, shows toast on success
- **Delete** button → confirmation dialog with toggles:
  - "Also delete GGUF file" (default false)
  - "Also delete MTP head file" (shown only if `mtp-head` is set, default false)
- If the backend response includes `shared_files`, show a second confirmation listing the referencing models: "This GGUF is still used by: model-a, model-b. Delete anyway?" with a "Force delete" action (re-sends with `?force=true`)
- Warning banner if model is currently in the `active` list: "This model is currently running. Changes may require a restart."

*GGUF file management* (tab or collapsible section below the editor):
- Table: filename, human-readable size, status ("active" — referenced by N model(s); "orphaned" — unused), actions
- Upload button → file picker → `POST /registry/upload`. Shows loading spinner during upload, error message on failure, auto-refreshes the list on success (following plugins page's styling for loading/error states)
- Orphaned files: "Delete" button per file; "Delete all orphaned" bulk action with confirmation
- Active files: shows which model(s) reference them (read-only)

**Page states**: Loading (skeleton cards), error (message + retry button), empty registry ("No models registered. Upload a GGUF file or add a model entry to get started."), no selection ("Select a model from the list or add a new one"), saving (inline spinner on Save button)

### Validation

Client-side (before allowing the Save action):
- `file` required and non-empty
- `ctx-size`, `predict`, `parallel`, `n-gpu-layers`, `image-min-tokens`, `image-max-tokens`: positive integers
- `cache-type-k`, `cache-type-v`: one of `q8_0`, `q4_0`, `turbo4`, `f16`
- `cost.input`, `cost.output`, `cost.cache_read`: non-negative numbers
- `extra-args`: must be an array of strings (or empty array `[]`)
- `speculative-type === "mtp"` → `mtp-head` required
- `speculative-type === "dflash"` → draft model file required (validates `spec-draft-model` field exists and is non-empty)
- `pflash === true` → `drafter` required

Backend: The `_validate_model_config()` helper in the endpoint handler checks the same rules on the server side (defense-in-depth). After `add()` or `update()` succeeds, `registry.validate()` runs deeper checks (file existence, DFlash GGUF contract validation, GPU range checks) and returns the result to the frontend so the user sees any backend-level issues immediately.

### Known Limitations

- **Concurrent edits**: Last-write-wins. The registry uses `mtime`-based reload with an `RLock`, but there's no per-model ETag or versioning. Same limitation as CLI workflow; made more visible by webui concurrency. Rare race: if model A is deleted between `add()` (ValueError because it existed) and `update()`, the endpoint returns 500. Acceptable given existing limitations.
- **No "rename model" operation**: The editor doesn't support renaming. Delete the old entry and create a new one with the correct name.
- **family_defaults are read-only**: Displayed as hints but not editable. A follow-up DEC can add editing capability (requiring a new registry method and endpoint).
- **Extra-args hints**: The flat-list editor provides no contextual guidance for which args go with which speculative type. Future UX improvement.

## Consequences

- Operator can manage models from the web UI without SSH
- Orphaned GGUF files become visible and garbage-collectible
- `family_defaults` becomes visible (read-only) in the UI for the first time
- New dependencies: none (uses existing `@lucide/svelte` `Boxes` icon, shadcn-svelte primitives, existing `apiFetch` pattern except for the upload path which uses raw `fetch`)
- The `/registry` endpoint exposes `models.json` content to any authenticated API user (read-only via `GET`; write endpoints also use `require_api`, so any valid API key can mutate — same trust boundary as plugin toggling)
- Future extensions: GPU pinning visualization, draft model management (pflash/dflash), model ingest from URLs (surfacing the existing `/ingest` endpoint), `family_defaults` editing
