/**
 * Per-session prompt drafts in localStorage. The session layout remounts the
 * composer whenever the session changes (`key={id}`), and a reload discards
 * React state outright; persisting the draft keeps typed text across both.
 * Attachments are deliberately not draftable — they hold live File handles
 * that cannot survive a remount or reload.
 */

const KEY_PREFIX = "session-prompt-draft:";
/** Keeps a runaway paste from turning every keystroke into a quota error. */
const MAX_DRAFT_CHARS = 100_000;

export function readSessionDraft(sessionId: string): string {
  try {
    return localStorage.getItem(KEY_PREFIX + sessionId) ?? "";
  } catch {
    return "";
  }
}

export function writeSessionDraft(sessionId: string, value: string): void {
  try {
    if (value.length === 0) {
      localStorage.removeItem(KEY_PREFIX + sessionId);
    } else {
      localStorage.setItem(KEY_PREFIX + sessionId, value.slice(0, MAX_DRAFT_CHARS));
    }
  } catch {
    // Storage is optional; the in-memory draft still works without it.
  }
}
