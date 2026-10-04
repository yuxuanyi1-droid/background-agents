// @vitest-environment jsdom
/// <reference types="@testing-library/jest-dom" />

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import * as matchers from "@testing-library/jest-dom/matchers";
import { ThinkingDisplaySelector } from "./thinking-display-selector";

expect.extend(matchers);

afterEach(cleanup);

describe("ThinkingDisplaySelector", () => {
  it("shows the selected mode on the trigger", () => {
    render(<ThinkingDisplaySelector value="summary" onChange={vi.fn()} />);

    const trigger = screen.getByRole("button", { name: "Thinking display: Summary" });
    expect(trigger).toHaveTextContent("Thinking:Summary");
  });

  it("offers the three display modes with the current one checked", async () => {
    render(<ThinkingDisplaySelector value="full" onChange={vi.fn()} />);

    fireEvent.pointerDown(screen.getByRole("button", { name: "Thinking display: Full" }), {
      button: 0,
      ctrlKey: false,
    });

    const options = await screen.findAllByRole("menuitemradio");
    expect(options.map((option) => option.textContent)).toEqual([
      "SummaryCollapsed, with a one-line preview",
      "FullExpanded, with the whole trail",
      "HiddenOnly the thinking indicator",
    ]);
    expect(screen.getByRole("menuitemradio", { name: /^Full/ })).toHaveAttribute(
      "aria-checked",
      "true"
    );
  });

  it("reports the chosen mode", async () => {
    const onChange = vi.fn();
    render(<ThinkingDisplaySelector value="summary" onChange={onChange} />);

    fireEvent.pointerDown(screen.getByRole("button", { name: "Thinking display: Summary" }), {
      button: 0,
      ctrlKey: false,
    });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: /^Hidden/ }));

    expect(onChange).toHaveBeenCalledWith("hidden");
  });
});
