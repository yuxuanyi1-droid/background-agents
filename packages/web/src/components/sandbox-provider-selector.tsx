"use client";

import type { SandboxProviderOption } from "@/hooks/use-sandbox-providers";
import type { SandboxProviderName } from "@open-inspect/shared/types/integrations";
import { BoxIcon } from "@/components/ui/icons";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

/** Which sandbox backend the next session runs on; shown only when the
 * deployment has more than one configured. */
export function SandboxProviderSelector({
  options,
  value,
  defaultValue,
  onChange,
  disabled,
}: {
  options: SandboxProviderOption[];
  value: SandboxProviderName;
  defaultValue: SandboxProviderName;
  onChange: (provider: SandboxProviderName) => void;
  disabled?: boolean;
}) {
  const label = options.find((option) => option.name === value)?.label ?? value;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className="flex max-w-full items-center gap-1.5 text-sm text-muted-foreground transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={`Sandbox provider: ${label}`}
        >
          <BoxIcon className="size-3.5" />
          <span className="max-w-28 truncate">{label}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuLabel>Sandbox provider</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={value}>
          {options.map((option) => (
            <DropdownMenuRadioItem
              key={option.name}
              value={option.name}
              onSelect={() => onChange(option.name)}
            >
              {option.label}
              {option.name === defaultValue ? (
                <span className="ml-auto pl-4 text-xs text-text-muted">default</span>
              ) : null}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
