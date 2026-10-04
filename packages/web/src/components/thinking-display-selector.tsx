"use client";

import { ChevronDownIcon } from "@/components/ui/icons";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { isThinkingDisplay, type ThinkingDisplay } from "@/lib/thinking-display";

const THINKING_DISPLAY_OPTIONS: Array<{
  value: ThinkingDisplay;
  label: string;
  description: string;
}> = [
  { value: "summary", label: "Summary", description: "Collapsed, with a one-line preview" },
  { value: "full", label: "Full", description: "Expanded, with the whole trail" },
  { value: "hidden", label: "Hidden", description: "Only the thinking indicator" },
];

export function ThinkingDisplaySelector({
  value,
  onChange,
}: {
  value: ThinkingDisplay;
  onChange: (value: ThinkingDisplay) => void;
}) {
  const selected =
    THINKING_DISPLAY_OPTIONS.find((option) => option.value === value) ??
    THINKING_DISPLAY_OPTIONS[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className="flex items-center gap-1.5 text-sm text-muted-foreground transition hover:text-foreground"
          aria-label={`Thinking display: ${selected.label}`}
        >
          <span className="shrink-0">Thinking:</span>
          <span className="text-secondary-foreground">{selected.label}</span>
          <ChevronDownIcon className="size-3.5 shrink-0 text-secondary-foreground" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="start" collisionPadding={8} className="w-64">
        <DropdownMenuRadioGroup
          value={value}
          onValueChange={(next) => {
            if (isThinkingDisplay(next)) onChange(next);
          }}
        >
          {THINKING_DISPLAY_OPTIONS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value}>
              <span className="min-w-0">
                <span className="block truncate">{option.label}</span>
                <span className="block truncate text-xs text-secondary-foreground">
                  {option.description}
                </span>
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
