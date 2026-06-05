// /ring view test (F4.1): renders the scrubbable ring player. hls.js needs MSE
// which jsdom lacks, so the useHls hook no-ops here; the test asserts the view
// renders its video element + description rather than actual playback.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Ring } from "./Ring";

describe("ring view (F4.1)", () => {
  it("renders the ring buffer player and explanation", () => {
    render(<Ring />);
    expect(screen.getByText("Fessel — Ring buffer")).toBeTruthy();
    expect(screen.getByText(/last few minutes of footage/)).toBeTruthy();
    expect(document.querySelector("video")).toBeTruthy();
  });
});
