// Minimal typings for radix-ui aggregate package
import type { ComponentType, PropsWithChildren } from "react";
type RadixComp = ComponentType<PropsWithChildren<Record<string, unknown>>>;

declare module "radix-ui" {
  export const Dialog: {
    Root: RadixComp;
    Trigger: RadixComp;
    Portal: RadixComp;
    Overlay: RadixComp;
    Content: RadixComp;
    Title: RadixComp;
    Close: RadixComp;
  };
  export const DropdownMenu: Record<string, RadixComp>;
  export const Tooltip: Record<string, RadixComp>;
  export const ToggleGroup: {
    Root: RadixComp;
    Item: RadixComp;
  };
  export const Switch: {
    Root: RadixComp;
    Thumb: RadixComp;
  };
  export const Slider: {
    Root: RadixComp;
    Track: RadixComp;
    Range: RadixComp;
    Thumb: RadixComp;
  };
  export const Checkbox: {
    Root: RadixComp;
    Indicator: RadixComp;
  };
}

