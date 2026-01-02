import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Monorepo/workspace guardrail:
  // When multiple lockfiles exist above this directory, Next.js (Turbopack)
  // may infer the wrong workspace root, which can lead to missing env vars
  // and even client code not hydrating as expected.
  //
  // Force Turbopack to treat `arxigraph/` as the root of the Next.js app.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
