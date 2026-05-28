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

## Verification

- `npm run test:unit -- --run tests/unit/agentic-sections.test.ts`
- `npm run build`

## Follow-Up

- Add fixture/e2e coverage for a provider that returns data URI image and text artifacts.
- Consider richer artifact metadata once backend tools can report filenames directly instead of deriving them from MIME type and index.
