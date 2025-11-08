"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import useSWR from "swr";
import { useParams } from "next/navigation";
import { getPaper } from "@/lib/api";
import type { ArtifactNode, PaperResponse } from "@/lib/types";
import D3GraphView from "@/components/D3GraphView";
import { unescapeLatex } from "@/lib/latex";
import { sanitizePreview } from "@/lib/sanitize";
import Logo from "@/components/Logo";
import Link from "next/link";

function DefinitionBankView({ bank }: { bank: PaperResponse["definition_bank"] }) {
    const entries = useMemo(() => {
        if (!bank) return [];
        return Object.entries(bank).map(([key, v]) => ({ key, ...v }));
    }, [bank]);

    const rootRef = useRef<HTMLDivElement | null>(null);
    const [open, setOpen] = useState<Record<string, boolean>>({});
    useEffect(() => {
        if (rootRef.current) {
            import("@/lib/katex").then((m) => m.renderKatex(rootRef.current!));
        }
    }, [entries, open]);

    if (!bank) {
        return <div className="text-sm text-gray-500">No definition bank available.</div>;
    }

    return (
        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1 text-black" ref={rootRef}>
            {entries.map((e) => {
                const defId = `defn-${e.key.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
                const isOpen = !!open[e.key];
                return (
                    <div key={e.key} className="border rounded p-2">
                        <button
                            type="button"
                            className="w-full text-left text-sm font-semibold flex items-center justify-between"
                            onClick={() => setOpen((o) => ({ ...o, [e.key]: !o[e.key] }))}
                            aria-expanded={isOpen}
                            aria-controls={defId}
                        >
                            <span>{e.term}</span>
                            <span className="ml-2 text-xs">{isOpen ? "▾" : "▸"}</span>
                        </button>

                        {isOpen ? (
                            <div id={defId} className="mt-2">
                                {e.aliases?.length ? (
                                    <div className="text-xs">Aliases: {e.aliases.join(", ")}</div>
                                ) : null}
                                <div
                                    className="text-sm mt-1 max-h-40 overflow-y-auto pr-1"
                                    dangerouslySetInnerHTML={{ __html: unescapeLatex(e.definition_text) }}
                                />
                                {e.dependencies?.length ? (
                                    <div className="text-xs mt-1">
                                        Depends on: {e.dependencies.join(", ")}
                                    </div>
                                ) : null}
                            </div>
                        ) : null}
                    </div>
                );
            })}
        </div>
    );
}

function ArtifactDetails({ node }: { node: ArtifactNode | null }) {
    const detailsRef = useRef<HTMLDivElement | null>(null);
    useEffect(() => {
        if (detailsRef.current) {
            import("@/lib/katex").then((m) => m.renderKatex(detailsRef.current!));
        }
    }, [node]);

    // Local UI state for collapsible proof section
    const [proofOpen, setProofOpen] = useState<boolean>(false);

    if (!node) {
        return <div className="text-sm text-gray-500">Select a node to see details.</div>;
    }

    return (
        <div className="space-y-3 artifact-details-root" ref={detailsRef} style={{ color: "#000" }}>
            <div>
                <div className="text-sm font-semibold">{node.display_name}</div>
                {node.label && <div className="text-xs text-gray-600">Label: {node.label}</div>}
                <div className="text-xs text-gray-500">
                    Lines: {node.position?.line_start}
                    {node.position?.line_end ? `–${node.position.line_end}` : ""}
                </div>
            </div>
            <div>
                <div className="text-xs uppercase text-gray-500 mb-1">Preview</div>
                <div
                    className="text-sm prose max-w-none"
                    // Use the full `content` (never truncate). First unescape JSON-escaped LaTeX, then sanitize.
                    // We prefer the complete `node.content` to avoid showing shortened `content_preview`.
                    dangerouslySetInnerHTML={{ __html: sanitizePreview(unescapeLatex(node.content)) }}
                />
            </div>
            {node.prerequisites_preview ? (
                <div>
                    <div className="text-xs uppercase text-gray-500 mb-1">Prerequisites</div>
                    <div
                        className="text-sm prose max-w-none"
                        // For prerequisites we render the available preview (no nonexistent `prerequisites` prop).
                        dangerouslySetInnerHTML={{ __html: sanitizePreview(unescapeLatex(node.prerequisites_preview || "")) }}
                    />
                </div>
            ) : null}
            {node.proof ? (
                <div>
                    <div className="flex items-center justify-between">
                        <div className="text-xs uppercase text-gray-500 mb-1">Proof</div>
                        <button
                            className="text-xs px-2 py-1 bg-gray-100 rounded"
                            onClick={() => setProofOpen((s) => !s)}
                            aria-expanded={proofOpen}
                            aria-controls="proof-content"
                        >
                            {proofOpen ? "Hide" : "Show"}
                        </button>
                    </div>
                    {proofOpen ? (
                        <div id="proof-content" className="mt-2">
                            <pre className="text-sm whitespace-pre-wrap bg-gray-50 p-2 rounded">{node.proof}</pre>
                        </div>
                    ) : null}
                </div>
            ) : null}
        </div>
    );
}

export default function PaperPage() {
    const params = useParams<{ arxiv_id: string }>();
    const arxivId = params.arxiv_id;

    const { data, error, isLoading } = useSWR<PaperResponse>(
        arxivId ? `/papers/${arxivId}` : null,
        () => getPaper(arxivId)
    );

    const [selected, setSelected] = useState<ArtifactNode | null>(null);

    useEffect(() => {
        // Reset when paper changes
        setSelected(null);
    }, [arxivId]);

    if (isLoading) {
        return (
            <>
                <header className="sticky top-0 z-40 bg-white border-b text-black">
                    <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4 justify-start">
                        <Link href="/" className="inline-flex items-center gap-2 leading-none shrink-0">
                            <Logo className="h-12 w-auto" withText={true} />
                        </Link>
                        <span className="text-xs sm:text-sm text-slate-600">Loading…</span>
                    </div>
                </header>
                <main className="min-h-screen p-6 md:p-12 bg-gradient-to-b from-slate-50 to-white">
                    <div className="max-w-5xl mx-auto">
                        <div className="text-sm text-gray-600">Loading paper {arxivId}…</div>
                    </div>
                </main>
            </>
        );
    }

    if (error || !data) {
        return (
            <>
                <header className="sticky top-0 z-40 bg-white border-b text-black">
                    <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4 justify-start">
                        <Link href="/" className="inline-flex items-center gap-2 leading-none shrink-0">
                            <Logo className="h-12 w-auto" withText={true} />
                        </Link>
                    </div>
                </header>
                <main className="min-h-screen p-6 md:p-12 bg-gradient-to-b from-slate-50 to-white">
                    <div className="max-w-5xl mx-auto">
                        <div className="text-sm text-red-600">
                            Failed to load paper: {(error as any)?.message || "Unknown error"}
                        </div>
                    </div>
                </main>
            </>
        );
    }

    const { graph, definition_bank } = data;

    return (
        <>
            <header className="sticky top-0 z-40 bg-white border-b text-black">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4 justify-start">
                    <Link href="/" className="inline-flex items-center gap-2 leading-none shrink-0">
                        <Logo className="h-12 w-auto" withText={true} />
                    </Link>
                </div>
            </header>
            <main className="min-h-screen p-6 md:p-12 bg-gradient-to-b from-slate-50 to-white">
                <div className="max-w-5xl mx-auto">
                    <div className="flex items-center gap-2 mb-6">
                        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
                            <a
                                href={`https://arxiv.org/abs/${arxivId}`}
                                target="_blank"
                                rel="noreferrer"
                                className="hover:underline underline-offset-4"
                                title={`Open ${arxivId} on arXiv`}
                            >
                                Paper {arxivId}
                            </a>
                        </h1>
                        <div className="text-sm text-slate-600">
                            Artifacts: {graph.stats?.node_count ?? graph.nodes.length} · Edges:{" "}
                            {graph.stats?.edge_count ?? graph.edges.length} · Mode: {graph.extractor_mode}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                        <div className="lg:col-span-8 bg-white rounded-xl ring-1 ring-slate-200 p-2 md:p-3 shadow-sm">
                            <D3GraphView graph={graph} onSelectNode={setSelected} height="68vh" />
                        </div>

                        <div className="lg:col-span-4 flex flex-col gap-4">
                            <div className="bg-white rounded-xl ring-1 ring-slate-200 p-3 shadow-sm">
                                <div className="text-sm font-semibold mb-2 text-black">Artifact Details</div>
                                <ArtifactDetails node={selected} />
                            </div>

                            <div className="bg-white rounded-xl ring-1 ring-slate-200 p-3 shadow-sm">
                                <div className="text-sm font-semibold mb-2 text-black">Definition Bank</div>
                                <DefinitionBankView bank={definition_bank} />
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
}
