// @vitest-environment jsdom
/// <reference types="@testing-library/jest-dom" />

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import * as matchers from "@testing-library/jest-dom/matchers";
import {
  DEFAULT_KEYBOARD_SHORTCUTS,
  type KeyboardShortcutBinding,
} from "@open-inspect/shared/types/keyboard-shortcuts";
import { usePromptInput } from "./use-prompt-input";

expect.extend(matchers);

const mocks = vi.hoisted(() => ({
  sendPrompt: vi.fn(),
  sendTyping: vi.fn(),
  clearAttachments: vi.fn(),
  uploadAll: vi.fn(),
}));

vi.mock("@/hooks/use-session-attachments", () => ({
  DEFAULT_ATTACHMENT_ONLY_MESSAGE: "See the attached files.",
  useSessionAttachments: () => ({
    attachments: [],
    attachmentError: null,
    isUploading: false,
    addFiles: vi.fn(),
    removeAttachment: vi.fn(),
    clearAttachments: mocks.clearAttachments,
    hasAttachments: () => false,
    uploadAll: mocks.uploadAll,
  }),
}));

function PromptHarness({
  canSubmit,
  sessionId = "session-1",
  sendShortcut = DEFAULT_KEYBOARD_SHORTCUTS["send-prompt"],
}: {
  canSubmit: boolean;
  sessionId?: string;
  sendShortcut?: KeyboardShortcutBinding;
}) {
  const prompt = usePromptInput(
    sessionId,
    mocks.sendPrompt,
    mocks.sendTyping,
    "model-1",
    undefined,
    false,
    "active",
    canSubmit,
    sendShortcut
  );

  return (
    <textarea
      aria-label="Prompt"
      value={prompt.prompt}
      onChange={prompt.handleInputChange}
      onKeyDown={prompt.handleKeyDown}
    />
  );
}

beforeEach(() => {
  mocks.sendPrompt.mockReset();
  mocks.sendTyping.mockReset();
  mocks.clearAttachments.mockReset();
  mocks.uploadAll.mockReset();
  localStorage.clear();
});

afterEach(cleanup);

describe("usePromptInput", () => {
  it("accepts draft edits but blocks the send shortcut before the session is ready", () => {
    render(<PromptHarness canSubmit={false} />);

    const input = screen.getByRole("textbox", { name: "Prompt" });
    fireEvent.change(input, { target: { value: "Draft while connecting" } });
    expect(input).toHaveValue("Draft while connecting");

    fireEvent.keyDown(input, { key: "Enter", ctrlKey: true });

    expect(mocks.sendPrompt).not.toHaveBeenCalled();
    expect(input).toHaveValue("Draft while connecting");
  });

  it("submits with the configured send shortcut instead of the default", () => {
    mocks.sendPrompt.mockResolvedValue({ ok: true });
    render(
      <PromptHarness
        canSubmit
        sendShortcut={{ code: "KeyJ", primary: false, alt: true, shift: false }}
      />
    );
    const input = screen.getByRole("textbox", { name: "Prompt" });
    fireEvent.change(input, { target: { value: "Ship it" } });

    fireEvent.keyDown(input, { key: "Enter", code: "Enter", ctrlKey: true });
    expect(mocks.sendPrompt).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: "j", code: "KeyJ", altKey: true });
    expect(mocks.sendPrompt).toHaveBeenCalledOnce();
  });

  it.each([
    ["Enter", false],
    ["Shift+Enter", true],
  ])("submits with %s when configured", (_label, shiftKey) => {
    mocks.sendPrompt.mockResolvedValue({ ok: true });
    render(
      <PromptHarness
        canSubmit
        sendShortcut={{ code: "Enter", primary: false, alt: false, shift: shiftKey }}
      />
    );
    const input = screen.getByRole("textbox", { name: "Prompt" });
    fireEvent.change(input, { target: { value: "Ship it" } });

    fireEvent.keyDown(input, { key: "Enter", code: "Enter", shiftKey });

    expect(mocks.sendPrompt).toHaveBeenCalledOnce();
  });

  it("restores the session's stored draft after hydration and persists edits", async () => {
    localStorage.setItem("session-prompt-draft:session-1", "Half-finished thought");
    render(<PromptHarness canSubmit={false} />);

    const input = screen.getByRole("textbox", { name: "Prompt" });
    await waitFor(() => expect(input).toHaveValue("Half-finished thought"));

    fireEvent.change(input, { target: { value: "Edited thought" } });
    expect(localStorage.getItem("session-prompt-draft:session-1")).toBe("Edited thought");
  });

  it("keeps drafts per session: another session's draft does not leak in", () => {
    localStorage.setItem("session-prompt-draft:session-2", "Other session");
    render(<PromptHarness canSubmit={false} sessionId="session-1" />);

    const input = screen.getByRole("textbox", { name: "Prompt" });
    expect(input).toHaveValue("");
  });

  it("clears the stored draft once the prompt is submitted", async () => {
    mocks.sendPrompt.mockResolvedValue({ ok: true });
    render(
      <PromptHarness
        canSubmit
        sendShortcut={{ code: "Enter", primary: true, alt: false, shift: false }}
      />
    );
    const input = screen.getByRole("textbox", { name: "Prompt" });
    fireEvent.change(input, { target: { value: "Ship it" } });
    expect(localStorage.getItem("session-prompt-draft:session-1")).toBe("Ship it");

    fireEvent.keyDown(input, { key: "Enter", code: "Enter", ctrlKey: true });

    await waitFor(() => expect(mocks.sendPrompt).toHaveBeenCalledOnce());
    await waitFor(() => expect(input).toHaveValue(""));
    expect(localStorage.getItem("session-prompt-draft:session-1")).toBeNull();
  });
});
