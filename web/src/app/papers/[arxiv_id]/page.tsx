"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import useSWR from "swr";
import { useParams, useSearchParams } from "next/navigation";
import { getPaper } from "@/lib/api";
import type { ArtifactNode, PaperResponse, DocumentGraph, DefinitionBank } from "@/lib/types";
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
        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1 text-[var(--text)]" ref={rootRef}>
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
        <div className="space-y-3 artifact-details-root text-[var(--text)]" ref={detailsRef}>
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

    const { data, error, isLoading, mutate } = useSWR<PaperResponse>(
        arxivId ? `/papers/${arxivId}` : null,
        () => getPaper(arxivId)
    );

    // Live streaming state
    const [liveGraph, setLiveGraph] = useState<DocumentGraph | null>(null);
    const [liveBank, setLiveBank] = useState<DefinitionBank | null>(null);
    const [streaming, setStreaming] = useState(false);
    const [stage, setStage] = useState<string | null>(null);
    const [depProg, setDepProg] = useState<{ processed: number; total: number } | null>(null);
    const linksRef = useRef<Record<string, Set<string>>>({});
    const esRef = useRef<EventSource | null>(null);
    const bankRef = useRef<DefinitionBank | null>(null);
    const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";
    const searchParams = useSearchParams();

    useEffect(() => {
        setLiveGraph(data?.graph || null);
        setLiveBank(data?.definition_bank || null);
        bankRef.current = data?.definition_bank || null;
        linksRef.current = {};
    }, [data]);

    useEffect(() => {
        bankRef.current = liveBank;
    }, [liveBank]);

    useEffect(() => {
        // Cleanup on unmount
        return () => {
            try {
                esRef.current?.close();
            } catch { }
            esRef.current = null;
        };
    }, []);

    // Auto-start streaming based on URL params (?stream=1&infer=true&enrich=true&force=true)
    useEffect(() => {
        if (!searchParams) return;
        const shouldStream = searchParams.get("stream") === "1";
        const inferParam = searchParams.get("infer") === "true";
        const enrichParam = searchParams.get("enrich") === "true";
        const forceParam = searchParams.get("force") === "true";
        if (shouldStream && !streaming && !esRef.current) {
            startStream({ infer: inferParam, enrich: enrichParam, force: forceParam });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [arxivId, searchParams]);

    const startStream = (opts: { infer: boolean; enrich: boolean; force?: boolean }) => {
        try {
            esRef.current?.close();
        } catch { }
        const url = `${API_BASE}/api/v1/papers/${arxivId}/stream-build?infer_dependencies=${opts.infer}&enrich_content=${opts.enrich}&force=${Boolean(opts.force)}`;
        const es = new EventSource(url);
        esRef.current = es;
        setStreaming(true);
        setStage("starting");

        es.addEventListener("nodes_seeded", (ev: MessageEvent) => {
            try {
                const payload = JSON.parse(ev.data);
                if (payload?.graph) {
                    setLiveGraph(payload.graph);
                }
            } catch { }
        });

        es.addEventListener("term_inferred", (ev: MessageEvent) => {
            try {
                const p = JSON.parse(ev.data);
                setLiveBank((prev) => {
                    const next: DefinitionBank = { ...(prev || {}) };
                    const key = p.term as string;
                    next[key] = {
                        term: p.term,
                        aliases: p.aliases || [],
                        definition_text: p.definition_text || "",
                        source_artifact_id: p.source_artifact_id || "",
                        dependencies: p.dependencies || [],
                    };
                    return next;
                });
            } catch { }
        });

        es.addEventListener("prerequisite_link", (ev: MessageEvent) => {
            try {
                const p = JSON.parse(ev.data) as { artifact_id: string; term: string };
                const map = linksRef.current;
                if (!map[p.artifact_id]) map[p.artifact_id] = new Set<string>();
                map[p.artifact_id].add(p.term);

                setLiveGraph((prev) => {
                    if (!prev) return prev;
                    const nodes = prev.nodes.map((n) => {
                        if (n.id !== p.artifact_id) return n;
                        const terms = Array.from(map[p.artifact_id] || []);
                        const bank = bankRef.current || {};
                        const defsHtml = terms
                            .map((t) => {
                                const def = (bank as any)[t];
                                const text = def?.definition_text || "";
                                return `<div><strong>${t}:</strong> ${text}</div>`;
                            })
                            .join("");
                        return { ...n, prerequisites_preview: defsHtml };
                    });
                    return { ...prev, nodes };
                });
            } catch { }
        });

        // New: stream dependency edges as soon as they are inferred
        es.addEventListener("dependency_edge", (ev: MessageEvent) => {
            try {
                const p = JSON.parse(ev.data) as {
                    source_id: string;
                    target_id: string;
                    dependency_type?: string | null;
                    dependency?: string | null;
                    context?: string | null;
                };
                setLiveGraph((prev) => {
                    if (!prev) return prev;
                    const exists = prev.edges.some(
                        (e) => e.source === p.source_id && e.target === p.target_id && (e.dependency_type || e.type) === (p.dependency_type || "generic_dependency")
                    );
                    if (exists) return prev;
                    const newEdge = {
                        source: p.source_id,
                        target: p.target_id,
                        context: p.context ?? null,
                        reference_type: null,
                        dependency_type: (p.dependency_type as any) ?? "generic_dependency",
                        dependency: p.dependency ?? null,
                        type: (p.dependency_type as any) ?? "generic_dependency",
                    };
                    return { ...prev, edges: [...prev.edges, newEdge] };
                });
            } catch { }
        });

        // New: dependency inference progress
        es.addEventListener("dependency_progress", (ev: MessageEvent) => {
            try {
                const p = JSON.parse(ev.data) as { processed: number; total: number };
                setDepProg(p);
            } catch { }
        });

        es.addEventListener("progress", (ev: MessageEvent) => {
            try {
                const p = JSON.parse(ev.data);
                setStage(p.stage || null);
            } catch { }
        });

        es.addEventListener("done", (_ev: MessageEvent) => {
            setStreaming(false);
            try {
                es.close();
            } catch { }
            esRef.current = null;
            mutate(); // refresh persisted snapshot from backend
        });

        es.onerror = () => {
            // Connection errors are expected on server restart; FastAPI will also send error/done
        };
    };

    const stopStream = () => {
        try {
            esRef.current?.close();
        } catch { }
        esRef.current = null;
        setStreaming(false);
    };

    const [selected, setSelected] = useState<ArtifactNode | null>(null);

    useEffect(() => {
        // Reset when paper changes
        setSelected(null);
    }, [arxivId]);

    if (isLoading) {
        return (
            <>
                <header className="sticky top-0 z-40 bg-[var(--surface)] border-b text-[var(--text)]">
                    <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4 justify-start">
                        <Link href="/" className="inline-flex items-center gap-2 leading-none shrink-0">
                            <Logo className="h-12 w-auto" withText={true} />
                        </Link>
                        <span className="text-xs sm:text-sm text-slate-600">Loading…</span>
                    </div>
                </header>
                <main className="min-h-screen p-6 md:p-12">
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
                <header className="sticky top-0 z-40 bg-[var(--surface)] border-b text-[var(--text)]">
                    <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4 justify-start">
                        <Link href="/" className="inline-flex items-center gap-2 leading-none shrink-0">
                            <Logo className="h-12 w-auto" withText={true} />
                        </Link>
                    </div>
                </header>
                <main className="min-h-screen p-6 md:p-12">
                    <div className="max-w-5xl mx-auto">
                        <div className="mb-4">
                            <div className="text-sm text-red-600">
                                Failed to load paper: {(error as any)?.message || "Unknown error"}
                            </div>
                            <div className="mt-3 flex items-center gap-3">
                                <button
                                    className="text-xs px-3 py-1 rounded bg-blue-600 text-white disabled:opacity-50"
                                    onClick={() => startStream({ infer: true, enrich: true })}
                                    disabled={streaming}
                                >
                                    {streaming ? "Streaming..." : "Start live build (deps + content)"}
                                </button>
                                <button
                                    className="text-xs px-3 py-1 rounded bg-gray-200 text-gray-800 disabled:opacity-50"
                                    onClick={stopStream}
                                    disabled={!streaming}
                                >
                                    Stop
                                </button>
                                {stage ? (
                                    <span className="text-xs text-slate-600">Stage: {stage}</span>
                                ) : null}
                            </div>
                        </div>

                        {liveGraph ? (
                            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                                <div className="lg:col-span-8 bg-[var(--surface)] rounded-xl ring-1 ring-slate-200 p-2 md:p-3 shadow-sm">
                                    <D3GraphView graph={liveGraph} onSelectNode={setSelected} height="68vh" />
                                </div>
                                <div className="lg:col-span-4 flex flex-col gap-4">
                                    <div className="bg-[var(--surface)] rounded-xl ring-1 ring-slate-200 p-3 shadow-sm">
                                        <div className="text-sm font-semibold mb-2 text-[var(--text)]">Artifact Details</div>
                                        <ArtifactDetails node={selected} />
                                    </div>
                                    <div className="bg-[var(--surface)] rounded-xl ring-1 ring-slate-200 p-3 shadow-sm">
                                        <div className="text-sm font-semibold mb-2 text-[var(--text)]">Definition Bank</div>
                                        <DefinitionBankView bank={liveBank ?? null} />
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-sm text-gray-600">
                                Not processed yet. Use the live build button above to stream the graph and definitions.
                            </div>
                        )}
                    </div>
                </main>
            </>
        );
    }

    const { graph, definition_bank } = data;

    return (
        <>
            <header className="sticky top-0 z-40 bg-[var(--surface)] border-b text-[var(--text)]">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4 justify-start">
                    <Link href="/" className="inline-flex items-center gap-2 leading-none shrink-0">
                        <Logo className="h-12 w-auto" withText={true} />
                    </Link>
                </div>
            </header>
            <main className="min-h-screen p-6 md:p-12">
                <div className="max-w-5xl mx-auto">
                    <div className="mb-6">
                        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
                            <a
                                href={`https://arxiv.org/abs/${arxivId}`}
                                target="_blank"
                                rel="noreferrer"
                                className="hover:underline underline-offset-4"
                                title={`Open ${arxivId} on arXiv`}
                            >
                                {data.title || `Paper ${arxivId}`}
                            </a>
                        </h1>
                        {data.authors && data.authors.length ? (
                            <div className="mt-1 text-sm text-slate-700">
                                {data.authors.join(", ")}
                            </div>
                        ) : null}
                        <div className="mt-2 text-sm text-slate-600">
                            Artifacts: {graph.stats?.node_count ?? graph.nodes.length} · Edges:{" "}
                            {graph.stats?.edge_count ?? graph.edges.length} · Mode: {graph.extractor_mode}
                        </div>

                        <div className="mt-3 flex items-center gap-3">
                            <button
                                className="text-xs px-3 py-1 rounded bg-blue-600 text-white disabled:opacity-50"
                                onClick={() => startStream({ infer: true, enrich: true })}
                                disabled={streaming}
                            >
                                {streaming ? "Streaming..." : "Start live build (deps + content)"}
                            </button>
                            <button
                                className="text-xs px-3 py-1 rounded bg-gray-200 text-gray-800 disabled:opacity-50"
                                onClick={stopStream}
                                disabled={!streaming}
                            >
                                Stop
                            </button>
                            <span className="text-xs text-slate-600">
                                {stage ? `Stage: ${stage}` : ""}
                                {depProg ? ` · Deps: ${depProg.processed}/${depProg.total}` : ""}
                            </span>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
                        <div className="lg:col-span-8 bg-[var(--surface)] rounded-xl ring-1 ring-slate-200 p-2 md:p-3 shadow-sm">
                            <D3GraphView graph={liveGraph ?? graph} onSelectNode={setSelected} height="68vh" />
                        </div>

                        <div className="lg:col-span-4 flex flex-col gap-4">
                            <div className="bg-[var(--surface)] rounded-xl ring-1 ring-slate-200 p-3 shadow-sm">
                                <div className="text-sm font-semibold mb-2 text-[var(--text)]">Artifact Details</div>
                                <ArtifactDetails node={selected} />
                            </div>

                            <div className="bg-[var(--surface)] rounded-xl ring-1 ring-slate-200 p-3 shadow-sm">
                                <div className="text-sm font-semibold mb-2 text-[var(--text)]">Definition Bank</div>
                                <DefinitionBankView bank={liveBank ?? definition_bank} />
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
}
