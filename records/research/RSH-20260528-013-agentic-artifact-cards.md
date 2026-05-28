# RSH-20260528-013: Agentic Artifact Cards
Opened: 2026-05-28 11-19-40 KST
Recorded by agent: Codex

## Question

Agentic tool outputs could attach images, but artifact visibility was still tied to expanded raw tool results. The product needs first-class artifact/file cards that reuse llama-ui's existing attachment preview path.

## Findings

- Tool results already persist attachments on tool-result messages through `DatabaseMessage.extra`.
- `ChatAttachmentsList` already provides thumbnails, file cards, and preview dialogs for stored message attachments.
- Agentic extraction only converted image data URIs into attachments; other data URI file types remained raw output.
- Expanded tool results rendered `[Attachment saved: ...]` marker lines, which are useful for the model but noisy for the user once cards exist.

## Change

- Added `ChatMessageAgenticArtifacts.svelte` for visible artifact cards beside agentic tool calls.
- Render artifact cards outside collapsed tool cards so users can see files without expanding raw output.
- Reused `ChatAttachmentsList` with left-aligned layout for agentic artifacts.
- Extended agentic data URI extraction to text, JSON, PDF, audio, and video attachments.
- Marked saved-attachment result lines so raw result rendering can hide those marker lines.

## Opencode Source Provenance

This was an adapted attachment/artifact behavior port; no opencode artifact UI or storage code was copied.

- Tool execute result attachment shape referenced `/home/yeowool/opencode/packages/opencode/src/tool/tool.ts:46-51`.
- Tool-result attachment normalization referenced `/home/yeowool/opencode/packages/opencode/src/session/tools.ts:94-102` for provider tool attachments and `/home/yeowool/opencode/packages/opencode/src/session/tools.ts:153-194` for MCP result-to-attachment conversion.
- Processor persistence of tool attachments referenced `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:168-190`, `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:280-293`, and `/home/yeowool/opencode/packages/opencode/src/session/processor.ts:455-500`.
- MessageV2 file/tool attachment schema and model conversion referenced `/home/yeowool/opencode/packages/opencode/src/session/message-v2.ts:160-168`, `/home/yeowool/opencode/packages/opencode/src/session/message-v2.ts:277-318`, and `/home/yeowool/opencode/packages/opencode/src/session/message-v2.ts:646-682`, `/home/yeowool/opencode/packages/opencode/src/session/message-v2.ts:790-809`.
- The webui artifact panel/header behavior is product-native; opencode supplied attachment semantics, not the right-side presentation panel.

## Verification

- `npm run test:unit -- --run tests/unit/agentic-sections.test.ts`
- `npm run build`

## Follow-Up

- Add fixture/e2e coverage for a provider that returns data URI image and text artifacts.
- Consider richer artifact metadata once backend tools can report filenames directly instead of deriving them from MIME type and index.
