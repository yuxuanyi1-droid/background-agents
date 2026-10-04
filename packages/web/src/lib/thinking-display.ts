/** The collapsed header shows at most this much of the segment's first line. */
export const THINKING_TEASER_MAX_CHARS = 120;

/**
 * The collapsed reasoning header's preview of one segment: its first
 * non-empty line, whitespace collapsed, truncated when it runs long.
 */
export function thinkingTeaser(content: string, maxChars = THINKING_TEASER_MAX_CHARS): string {
  const line = content.split("\n").find((candidate) => candidate.trim());
  if (!line) return "";

  const collapsed = line.trim().replace(/\s+/g, " ");
  return collapsed.length > maxChars ? `${collapsed.slice(0, maxChars).trimEnd()}…` : collapsed;
}
