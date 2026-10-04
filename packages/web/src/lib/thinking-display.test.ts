import { describe, expect, it } from "vitest";
import { THINKING_TEASER_MAX_CHARS, thinkingTeaser } from "./thinking-display";

describe("thinkingTeaser", () => {
  it("previews the first non-empty line with collapsed whitespace", () => {
    expect(thinkingTeaser("\n\n  Checking   the query\nplan later")).toBe("Checking the query");
  });

  it("keeps a short single-line segment verbatim", () => {
    expect(thinkingTeaser("Checking the query plan")).toBe("Checking the query plan");
  });

  it("truncates a long line to the default budget with an ellipsis", () => {
    expect(thinkingTeaser("a".repeat(500))).toBe(`${"a".repeat(THINKING_TEASER_MAX_CHARS)}…`);
  });

  it("truncates at a custom budget and drops the trailing space before the ellipsis", () => {
    expect(thinkingTeaser("one two three", 8)).toBe("one two…");
  });

  it("returns nothing for blank content", () => {
    expect(thinkingTeaser("")).toBe("");
    expect(thinkingTeaser("\n   \n")).toBe("");
  });
});
