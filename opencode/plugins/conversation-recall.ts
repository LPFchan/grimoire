import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"

type RecallRow = {
  part_id: string
  message_id: string
  session_id: string
  session_title: string | null
  role: string | null
  part_type: string | null
  tool_name: string | null
  tool_status: string | null
  tool_command: string | null
  time_created: number | null
  text: string | null
  tool_output: string | null
}

const SEARCH_EXCERPT_CHARS = 280
const DEFAULT_PART_TYPES = ["text", "reasoning"] as const

const sqlQuote = (value: string) => `'${value.replace(/'/g, "''")}'`

const clip = (value: string, max: number) => {
  if (value.length <= max) return value
  return `${value.slice(0, Math.max(0, max - 3))}...`
}

const collapseWhitespace = (value: string) => value.replace(/\s+/g, " ").trim()

const formatTime = (value: number | null) => {
  if (!value) return "unknown"

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toISOString()
}

const visibleContent = (row: RecallRow) => {
  if (row.part_type === "tool") return row.tool_output ?? ""
  return row.text ?? ""
}

const formatSearchRow = (row: RecallRow, index: number) => {
  const content = clip(collapseWhitespace(visibleContent(row)) || "(empty)", SEARCH_EXCERPT_CHARS)
  const header = [
    `${index + 1}. part ${row.part_id}`,
    `message ${row.message_id}`,
    row.role ?? "unknown-role",
    row.part_type === "tool" ? `tool:${row.tool_name ?? "unknown"}` : row.part_type ?? "unknown-type",
    formatTime(row.time_created),
  ].join(" | ")

  const session = row.session_title
    ? `session ${row.session_id} | ${row.session_title}`
    : `session ${row.session_id}`

  return `${header}\n${session}\n${content}`
}

const formatPartRow = (row: RecallRow, includeToolOutput: boolean) => {
  const lines = [
    `part: ${row.part_id}`,
    `message: ${row.message_id}`,
    `session: ${row.session_id}`,
    `session_title: ${row.session_title ?? ""}`,
    `role: ${row.role ?? "unknown"}`,
    `type: ${row.part_type ?? "unknown"}`,
    `time: ${formatTime(row.time_created)}`,
  ]

  if (row.part_type === "tool") {
    lines.push(`tool: ${row.tool_name ?? "unknown"}`)
    lines.push(`tool_status: ${row.tool_status ?? "unknown"}`)
    if (row.tool_command) lines.push(`tool_command: ${row.tool_command}`)
    lines.push("")
    if (includeToolOutput) {
      lines.push("output:")
      lines.push(row.tool_output ?? "(no inline tool output stored in part.state.output)")
    } else {
      lines.push("tool output omitted; rerun with includeToolOutput=true to retrieve it")
    }
    return lines.join("\n")
  }

  lines.push("")
  lines.push("text:")
  lines.push(row.text ?? "")
  return lines.join("\n")
}

const buildBaseSelect = (includeToolOutput: boolean) => `
select
  p.id as part_id,
  p.message_id,
  p.session_id,
  s.title as session_title,
  json_extract(m.data, '$.role') as role,
  json_extract(p.data, '$.type') as part_type,
  json_extract(p.data, '$.tool') as tool_name,
  json_extract(p.data, '$.state.status') as tool_status,
  json_extract(p.data, '$.state.input.command') as tool_command,
  p.time_created,
  json_extract(p.data, '$.text') as text,
  ${includeToolOutput ? "json_extract(p.data, '$.state.output')" : "null"} as tool_output
from part p
join message m on m.id = p.message_id
join session s on s.id = p.session_id
`

const buildSearchQuery = (input: {
  query: string
  sessionID?: string
  scope: "current" | "all"
  currentSessionID: string
  limit: number
  includeToolOutput: boolean
  includeReasoning: boolean
  role?: string
}) => {
  const textExpr = `coalesce(json_extract(p.data, '$.text'), '')`
  const toolOutputExpr = `coalesce(json_extract(p.data, '$.state.output'), '')`
  const queryText = sqlQuote(input.query)
  const partTypes = input.includeReasoning ? DEFAULT_PART_TYPES : (["text"] as const)
  const textTypeFilter = partTypes.map((type) => sqlQuote(type)).join(", ")

  const where = [
    input.sessionID
      ? `p.session_id = ${sqlQuote(input.sessionID)}`
      : input.scope === "current"
        ? `p.session_id = ${sqlQuote(input.currentSessionID)}`
        : "1 = 1",
    input.includeToolOutput
      ? `(
          (json_extract(p.data, '$.type') in (${textTypeFilter}) and instr(lower(${textExpr}), lower(${queryText})) > 0)
          or
          (json_extract(p.data, '$.type') = 'tool' and instr(lower(${toolOutputExpr}), lower(${queryText})) > 0)
        )`
      : `(json_extract(p.data, '$.type') in (${textTypeFilter}) and instr(lower(${textExpr}), lower(${queryText})) > 0)`,
  ]

  if (input.role) {
    const roleFilter = `json_extract(m.data, '$.role') = ${sqlQuote(input.role)}`
    where.push(`(${roleFilter})`)
  }

  return `${buildBaseSelect(input.includeToolOutput)} where ${where.join(" and ")} order by p.time_created desc limit ${input.limit}`
}

const buildListQuery = (input: {
  sessionID?: string
  scope: "current" | "all"
  currentSessionID: string
  limit: number
  includeToolOutput: boolean
  includeReasoning: boolean
  role?: string
}) => {
  const partTypes = input.includeReasoning ? DEFAULT_PART_TYPES : (["text"] as const)
  const typeFilter = partTypes.map((type) => sqlQuote(type)).join(", ")

  const where = [
    input.sessionID
      ? `p.session_id = ${sqlQuote(input.sessionID)}`
      : input.scope === "current"
        ? `p.session_id = ${sqlQuote(input.currentSessionID)}`
        : "1 = 1",
    `json_extract(p.data, '$.type') in (${typeFilter})`,
  ]

  if (input.role) {
    where.push(`json_extract(m.data, '$.role') = ${sqlQuote(input.role)}`)
  }

  return `${buildBaseSelect(input.includeToolOutput)} where ${where.join(" and ")} order by p.time_created asc limit ${input.limit}`
}

const buildPartQuery = (partID: string, includeToolOutput: boolean) =>
  `${buildBaseSelect(includeToolOutput)} where p.id = ${sqlQuote(partID)} limit 1`

const buildMessageQuery = (messageID: string, includeReasoning: boolean, includeToolOutput: boolean) => {
  const types = includeReasoning ? `'text', 'reasoning', 'tool'` : `'text', 'tool'`
  return `${buildBaseSelect(includeToolOutput)} where p.message_id = ${sqlQuote(messageID)} and json_extract(p.data, '$.type') in (${types}) order by p.time_created asc`
}

export const ConversationRecallPlugin: Plugin = async ({ $ }) => {
  const runQuery = async <Row>(sql: string) => {
    const result = await $`opencode --pure db ${sql} --format json`.quiet()
    return JSON.parse(String(result.stdout || "[]")) as Row[]
  }

  return {
    tool: {
      conversation_recall: tool({
        description:
           "Search, list, or fetch exact conversation content from OpenCode's local session log. Use this when older wording may have been compressed out of context and you need the original text back. Defaults to the current session and includes normal text plus reasoning for parity with OpenCode session replay.",
        args: {
          action: tool.schema
            .enum(["search", "get_part", "get_message", "list"])
            .describe("search by exact substring, fetch a specific part, fetch all visible parts for one message, or list parts chronologically"),
          role: tool.schema
            .string()
            .optional()
            .describe("Filter by message role (user, assistant, tool, system). Applies to search and list actions."),
          query: tool.schema
            .string()
            .optional()
            .describe("Exact substring to search for. Required when action=search."),
          partID: tool.schema
            .string()
            .optional()
            .describe("Part id to fetch exactly. Required when action=get_part."),
          messageID: tool.schema
            .string()
            .optional()
            .describe("Message id to fetch. Required when action=get_message."),
          sessionID: tool.schema
            .string()
            .optional()
            .describe("Override the session to search. Defaults to the current session."),
          scope: tool.schema
            .enum(["current", "all"])
            .optional()
            .describe("For search and list: current session or all sessions. Ignored when sessionID is set."),
          limit: tool.schema
            .number()
            .int()
            .min(1)
            .max(10)
            .optional()
            .describe("Maximum results to return. Default is 5."),
          includeToolOutput: tool.schema
            .boolean()
            .optional()
            .describe("Also search or include tool outputs. Default is false. Applies to search and get_message."),
          includeReasoning: tool.schema
            .boolean()
            .optional()
            .describe("Include reasoning parts. Default is true to match OpenCode session replay. Applies to search, list, and get_message."),
        },
        async execute(args, context) {
          const action = args.action
          const limit = Math.min(10, Math.max(1, args.limit ?? 5))
          const scope = args.scope ?? "current"
          const includeToolOutput = args.includeToolOutput ?? false
          const includeReasoning = args.includeReasoning ?? true

          if (action === "search") {
            if (!args.query?.trim()) {
              throw new Error("conversation_recall: query is required when action=search")
            }

            context.metadata({
              title: `Recall: ${clip(args.query.trim(), 48)}`,
              metadata: {
                action,
                query: args.query.trim(),
                role: args.role,
                scope: args.sessionID ? "session" : scope,
                includeReasoning,
              },
            })

            const rows = await runQuery<RecallRow>(
              buildSearchQuery({
                query: args.query.trim(),
                sessionID: args.sessionID,
                scope,
                currentSessionID: context.sessionID,
                limit,
                includeToolOutput,
                includeReasoning,
                role: args.role,
              }),
            )

            if (!rows.length) {
              const searchedScope = args.sessionID
                ? `session ${args.sessionID}`
                : scope === "all"
                  ? "all sessions"
                  : `current session ${context.sessionID}`

              return {
                output: `No matches found for ${JSON.stringify(args.query.trim())} in ${searchedScope}${args.role ? ` with role "${args.role}"` : ""}.`,
                metadata: {
                  action,
                  query: args.query.trim(),
                  role: args.role,
                  matches: 0,
                  includeReasoning,
                },
              }
            }

            const searchedScope = args.sessionID
              ? `session ${args.sessionID}`
              : scope === "all"
                ? "all sessions"
                : `current session ${context.sessionID}`

            const body = [
              `Search query: ${JSON.stringify(args.query.trim())}`,
              `Scope: ${searchedScope}`,
              `Role: ${args.role ?? "all roles"}`,
              `Include reasoning: ${includeReasoning ? "yes" : "no"}`,
              `Matches: ${rows.length}`,
              "",
              ...rows.map(formatSearchRow),
              "",
              'Use `action="get_part"` with a returned `partID` to fetch the exact full content.',
            ].join("\n")

            return {
              output: body,
              metadata: {
                action,
                query: args.query.trim(),
                role: args.role,
                matches: rows.length,
                sessionID: args.sessionID ?? (scope === "current" ? context.sessionID : null),
                includeReasoning,
              },
            }
          }

          if (action === "list") {
            const roleLabel = args.role ?? "all roles"
            context.metadata({
              title: `List: ${roleLabel}`,
              metadata: {
                action,
                role: args.role,
                scope: args.sessionID ? "session" : scope,
                includeReasoning,
              },
            })

            const rows = await runQuery<RecallRow>(
              buildListQuery({
                sessionID: args.sessionID,
                scope,
                currentSessionID: context.sessionID,
                limit,
                includeToolOutput,
                includeReasoning,
                role: args.role,
              }),
            )

            const listedScope = args.sessionID
              ? `session ${args.sessionID}`
              : scope === "all"
                ? "all sessions"
                : `current session ${context.sessionID}`

            if (!rows.length) {
              return {
                output: `No parts found in ${listedScope}${args.role ? ` with role "${args.role}"` : ""}.`,
                metadata: {
                  action,
                  role: args.role,
                  results: 0,
                },
              }
            }

            const body = [
              `Scope: ${listedScope}`,
              `Role: ${roleLabel}`,
              `Include reasoning: ${includeReasoning ? "yes" : "no"}`,
              `Results: ${rows.length}`,
              `Order: chronological (oldest first)`,
              "",
              ...rows.map(formatSearchRow),
              "",
              'Use `action="get_part"` with a returned `partID` to fetch the exact full content.',
            ].join("\n")

            return {
              output: body,
              metadata: {
                action,
                role: args.role,
                results: rows.length,
                sessionID: args.sessionID ?? (scope === "current" ? context.sessionID : null),
                includeReasoning,
              },
            }
          }

          if (action === "get_part") {
            if (!args.partID?.trim()) {
              throw new Error("conversation_recall: partID is required when action=get_part")
            }

            context.metadata({
              title: `Recall part: ${args.partID.trim()}`,
              metadata: {
                action,
                partID: args.partID.trim(),
              },
            })

            const [row] = await runQuery<RecallRow>(buildPartQuery(args.partID.trim(), includeToolOutput))

            if (!row) {
              return {
                output: `No part found for id ${args.partID.trim()}.`,
                metadata: {
                  action,
                  partID: args.partID.trim(),
                  found: false,
                },
              }
            }

            if (row.part_type !== "text" && row.part_type !== "reasoning" && row.part_type !== "tool") {
              return {
                output: `Part ${args.partID.trim()} exists but is type ${row.part_type ?? "unknown"}, which this tool does not expose.`,
                metadata: {
                  action,
                  partID: args.partID.trim(),
                  found: true,
                  allowed: false,
                },
              }
            }

            return {
              output: formatPartRow(row, includeToolOutput),
              metadata: {
                action,
                partID: args.partID.trim(),
                messageID: row.message_id,
                sessionID: row.session_id,
                role: row.role,
                type: row.part_type,
              },
            }
          }

          if (!args.messageID?.trim()) {
            throw new Error("conversation_recall: messageID is required when action=get_message")
          }

          context.metadata({
            title: `Recall message: ${args.messageID.trim()}`,
            metadata: {
              action,
              messageID: args.messageID.trim(),
              includeToolOutput,
              includeReasoning,
            },
          })

          const rows = await runQuery<RecallRow>(buildMessageQuery(args.messageID.trim(), includeReasoning, includeToolOutput))

          if (!rows.length) {
            return {
              output: `No visible parts found for message ${args.messageID.trim()}.`,
              metadata: {
                action,
                messageID: args.messageID.trim(),
                found: false,
                includeReasoning,
              },
            }
          }

          const first = rows[0]
          const sections = [
            `message: ${args.messageID.trim()}`,
            `session: ${first.session_id}`,
            `session_title: ${first.session_title ?? ""}`,
            `role: ${first.role ?? "unknown"}`,
            `include_reasoning: ${includeReasoning ? "yes" : "no"}`,
            `parts: ${rows.length}`,
            "",
          ]

          for (const row of rows) {
            sections.push(`part ${row.part_id} | ${row.part_type ?? "unknown"} | ${formatTime(row.time_created)}`)

            if (row.part_type === "tool") {
              sections.push(`tool: ${row.tool_name ?? "unknown"} | status: ${row.tool_status ?? "unknown"}`)
              if (row.tool_command) sections.push(`command: ${row.tool_command}`)
              if (includeToolOutput) {
                sections.push("output:")
                sections.push(row.tool_output ?? "(no inline tool output stored in part.state.output)")
              } else {
                sections.push("tool output omitted; rerun with includeToolOutput=true to retrieve it")
              }
            } else {
              sections.push(row.text ?? "")
            }

            sections.push("")
          }

          return {
            output: sections.join("\n").trimEnd(),
            metadata: {
              action,
              messageID: args.messageID.trim(),
              sessionID: first.session_id,
              role: first.role,
              parts: rows.length,
              includeReasoning,
            },
          }
        },
      }),
    },
  }
}
