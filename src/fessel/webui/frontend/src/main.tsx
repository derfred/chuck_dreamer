import React from "react";
import { createRoot } from "react-dom/client";
import { Live } from "./Live";

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <React.StrictMode>
      <Live />
    </React.StrictMode>,
  );
}
