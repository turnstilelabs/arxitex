"use client";

import { useEffect, useMemo, useState } from "react";

import { buildDistilledHtml, type DistillModel } from "@/components/constellations/distiller";

/**
 * Client-only distilled proof page.
 *
 * Flow:
 * - The graph view calls renderDistilledWindow(model), which stores the
 *   DistillModel in localStorage and navigates to this route.
 * - Here we read the stored model, build the full HTML shell using the
 *   same template as the original Constellations project, and write it
 *   into the document. The browser Back button then naturally returns to
 *   the /paper/[arxivId] graph page.
 */
export default function DistilledProofPage() {
    const [modelState] = useState<{ source: "name" | "localStorage" | null; model: DistillModel | null }>(() => {
        if (typeof window === "undefined") {
            return { source: null, model: null };
        }

        // Prefer `window.name` (more space, survives navigation).
        if (window.name) {
            try {
                const parsed = JSON.parse(window.name) as any;
                if (parsed && parsed.__arxigraphDistilledModel) {
                    return { source: "name", model: parsed.__arxigraphDistilledModel as DistillModel };
                }
            } catch {
                // ignore
            }
        }

        // Fallback to localStorage.
        try {
            const raw = window.localStorage.getItem("arxigraph:distilledModel");
            if (raw) {
                return { source: "localStorage", model: JSON.parse(raw) as DistillModel };
            }
        } catch {
            // ignore
        }

        return { source: null, model: null };
    });

    // Clear window.name once we've consumed it, but don't trigger re-renders.
    useEffect(() => {
        if (modelState.source === "name") {
            window.name = "";
        }
    }, [modelState.source]);

    const srcDoc = useMemo(() => {
        if (!modelState.model) return null;
        return buildDistilledHtml(modelState.model);
    }, [modelState.model]);

    if (!modelState.model) {
        return (
            <main style={{ padding: 16, fontFamily: "Inter, system-ui, sans-serif" }}>
                <h1 style={{ fontSize: 18, marginBottom: 8 }}>Distilled proof</h1>
                <p style={{ opacity: 0.85 }}>
                    No distilled proof data was found for this tab. Please go back to the graph page and click
                    “Generate Distilled Proof” again.
                </p>
            </main>
        );
    }

    if (!srcDoc) {
        return (
            <main style={{ padding: 16, fontFamily: "Inter, system-ui, sans-serif" }}>
                <p style={{ opacity: 0.85 }}>Loading distilled proof…</p>
            </main>
        );
    }

    // Render the distilled HTML inside an iframe.
    // This avoids overwriting the Next.js document (which breaks navigation/back),
    // while still giving the full distilled “page” experience.
    return (
        <iframe
            title="Distilled proof"
            srcDoc={srcDoc}
            // Avoid browser/devtools trying to resolve sourcemaps for about:srcdoc.
            // This also ensures all relative URLs inside the iframe resolve against our origin.
            src="/constellations/blank.html"
            style={{ width: "100%", height: "100vh", border: 0, display: "block" }}
        />
    );
}
