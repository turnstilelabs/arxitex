"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export type DefinitionBankEntry = {
    term: string;
    definitionText: string;
    aliases: string[];
};

interface DefinitionBankCardProps {
    definitions: DefinitionBankEntry[];
}

export default function DefinitionBankCard({ definitions }: DefinitionBankCardProps) {
    const [query, setQuery] = useState("");
    const containerRef = useRef<HTMLDivElement | null>(null);

    const renderDefinitionText = (raw: string): string => {
        if (!raw) return '';

        // Strip non-printable control characters that can sneak in from LLM output
        // (e.g. ANSI escape codes) and cause MathJax parse failures.
        const withoutControls = raw.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '');

        // The extracted LaTeX often contains paper-specific macros from the
        // LaTeX preamble (e.g. \F, \R). We don't have the preamble here, so
        // MathJax can't expand them.
        //
        // Best-effort fallback: turn single-letter macros like "\\F" into the
        // literal letter "F" so we render *something* rather than showing a
        // MathJax error.
        //
        // NOTE: we intentionally only strip *single uppercase letter* macros
        // to avoid breaking standard commands like \frac, \alpha, \mathbb, etc.
        return withoutControls
            .replace(/\\eps\b/g, '\\varepsilon')
            .replace(/\\([A-Z])(?![A-Za-z])/g, '$1');
    };

    const renderTerm = (raw: string): string => {
        if (!raw) return '';
        const withoutControls = raw.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '');
        // Same fallback behavior as definitions: strip unknown single-letter
        // uppercase macros so e.g. "\\F_1" becomes "F_1".
        return withoutControls.replace(/\\([A-Z])(?![A-Za-z])/g, '$1');
    };

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!q) return definitions;

        return definitions.filter((d) => {
            const haystack = `${d.term} ${d.definitionText}`.toLowerCase();
            return haystack.includes(q);
        });
    }, [definitions, query]);

    const totalCount = definitions.length;

    // MathJax is loaded globally in `src/app/layout.tsx`, but it does not
    // automatically typeset math inside client-rendered / dynamically updated
    // React content. Trigger a typeset pass whenever the visible content
    // changes (definitions list or search query).
    useEffect(() => {
        const mj = (typeof window !== "undefined" ? (window as any).MathJax : null) as any;
        if (!mj?.typesetPromise) return;
        if (!containerRef.current) return;

        // Typeset only within this card for performance.
        mj.typesetPromise([containerRef.current]).catch(() => {
            // best-effort: avoid breaking the UI if MathJax fails
        });
    }, [definitions, query]);

    return (
        <div
            ref={containerRef}
            className="w-full rounded-lg p-4 border"
            style={{
                background: "var(--surface1)",
                borderColor: "var(--border-color)",
                color: "var(--primary-text)",
            }}
        >
            <div className="flex items-baseline justify-between gap-2 flex-wrap">
                <div>
                    <h2 className="text-base font-semibold">Definition bank</h2>
                    <p
                        className="mt-1 text-xs"
                        style={{ color: "var(--secondary-text)" }}
                    >
                        Definitions extracted from the paper to help interpret the graph.
                    </p>
                </div>
                {totalCount > 0 && (
                    <div
                        className="text-xs font-medium"
                        style={{ color: "var(--secondary-text)" }}
                    >
                        {totalCount} term{totalCount === 1 ? "" : "s"}
                    </div>
                )}
            </div>

            {totalCount > 4 && (
                <div className="mt-3">
                    <input
                        type="search"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search definitions…"
                        className="w-full px-2 py-1.5 rounded-md text-sm border bg-transparent"
                        style={{
                            borderColor: "var(--border-color)",
                            color: "var(--primary-text)",
                        }}
                    />
                </div>
            )}

            <div
                className="mt-3 space-y-2 text-sm max-h-80 overflow-y-auto pr-1"
                style={{ color: "var(--secondary-text)" }}
            >
                {filtered.map((d) => (
                    <div
                        key={d.term}
                        className="flex flex-col gap-0.5 px-2 py-1 rounded-md hover:bg-[var(--surface2)]"
                    >
                        <div className="font-semibold" style={{ color: "var(--primary-text)" }}>
                            {renderTerm(d.term)}
                        </div>
                        <div className="text-xs leading-snug line-clamp-2">
                            {renderDefinitionText(d.definitionText) || "(No definition text available)"}
                        </div>
                    </div>
                ))}

                {filtered.length === 0 && (
                    <div className="text-xs" style={{ color: "var(--secondary-text)" }}>
                        No definitions match your search.
                    </div>
                )}
            </div>
        </div>
    );
}
