"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import useSWR from "swr";
import { useParams, useSearchParams } from "next/navigation";
import { getPaper, getPdfUrl, getAnchors } from "@/lib/api";
import type { ArtifactNode, PaperResponse, DocumentGraph, DefinitionBank, ArtifactAnchorIndex } from "@/lib/types";
import D3GraphView from "@/components/D3GraphView";
import { unescapeLatex } from "@/lib/latex";
import { sanitizePreview } from "@/lib/sanitize";
import Logo from "@/components/Logo";
import Link from "next/link";
import type { PaperReaderHandle } from "@/components/PaperReader";
import dynamic from "next/dynamic";
const PaperReader = dynamic(() => import("@/components/PaperReader"), { ssr: false });
import { PanelGroup, Panel, PanelResizeHandle } from "react-resizable-panels";
import { buildExportBundle, downloadBundleAsJson, downloadBundleAsGzip, suggestExportFilename } from "@/lib/export";

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

function ArtifactDetails({ node, bank }: { node: ArtifactNode | null; bank: DefinitionBank | null }) {
    const detailsRef = useRef<HTMLDivElement | null>(null);
    useEffect(() => {
        if (detailsRef.current) {
            import("@/lib/katex").then((m) => m.renderKatex(detailsRef.current!));
        }
    }, [node]);

    // Local UI state for collapsible proof section
    const [proofOpen, setProofOpen] = useState<boolean>(false);

    // Resolve prerequisite symbols to full definitions from the bank when possible
    const resolvedPrereqs = useMemo(() => {
        if (!node || !bank) return [] as Array<{ term: string; html: string }>;

        // Build a lookup of normalized term/aliases -> definition entry
        const norm = (s: string) => (s || "").toLowerCase().replace(/\s+/g, " ").trim();
        const entries = Object.values(bank as any) as Array<{
            term?: string;
            aliases?: string[];
            definition_text?: string;
        }>;

        const map: Record<string, { term: string; definition_text: string }> = {};
        for (const e of entries) {
            const t = e?.term || "";
            const def = e?.definition_text || "";
            if (t) map[norm(t)] = { term: t, definition_text: def };
            if (Array.isArray(e?.aliases)) {
                for (const a of e.aliases) {
                    if (!a) continue;
                    map[norm(a)] = { term: a, definition_text: def };
                }
            }
        }

        // Candidate prerequisite names
        const explicitList = (node as any).prerequisites as string[] | undefined;

        let candidates: string[] = [];
        if (Array.isArray(explicitList) && explicitList.length) {
            candidates = explicitList;
        } else if (typeof node.prerequisites_preview === "string" && node.prerequisites_preview) {
            // Extract plain text from preview and attempt to match known terms
            const plain = node.prerequisites_preview
                .replace(/<[^>]*>/g, " ") // strip tags
                .replace(/\s+/g, " ")
                .trim()
                .toLowerCase();

            // Pick those terms that actually appear in the preview text (simple contains heuristic)
            const seen = new Set<string>();
            for (const key of Object.keys(map)) {
                if (plain.includes(key) && !seen.has(key)) {
                    candidates.push(map[key].term);
                    seen.add(key);
                }
            }
        }

        // Build resolved list
        const out: Array<{ term: string; html: string }> = [];
        const added = new Set<string>();
        for (const cand of candidates) {
            const k = norm(cand);
            const hit = map[k];
            if (hit && !added.has(hit.term)) {
                out.push({
                    term: hit.term,
                    html: hit.definition_text || "",
                });
                added.add(hit.term);
            }
        }
        return out;
    }, [node, bank]);

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
            {resolvedPrereqs.length > 0 ? (
                <div>
                    <div className="text-xs uppercase text-gray-500 mb-1">Prerequisites</div>
                    <div className="space-y-2">
                        {resolvedPrereqs.map((p) => (
                            <div key={p.term} className="text-sm prose max-w-none">
                                <div className="font-semibold inline-block mr-1">{p.term}:</div>
                                <span
                                    dangerouslySetInnerHTML={{
                                        __html: sanitizePreview(unescapeLatex(p.html || "")),
                                    }}
                                />
                            </div>
                        ))}
                    </div>
                </div>
            ) : node.prerequisites_preview ? (
                <div>
                    <div className="text-xs uppercase text-gray-500 mb-1">Prerequisites</div>
                    <div
                        className="text-sm prose max-w-none"
                        dangerouslySetInnerHTML={{
                            __html: sanitizePreview(unescapeLatex(node.prerequisites_preview || "")),
                        }}
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
        () => getPaper(arxivId),
        { revalidateOnFocus: false, revalidateOnReconnect: false, refreshInterval: 0 }
    );

    // Live streaming state
    const [liveGraph, setLiveGraph] = useState<DocumentGraph | null>(null);
    const [liveBank, setLiveBank] = useState<DefinitionBank | null>(null);
    const [streaming, setStreaming] = useState(false);
    const [stage, setStage] = useState<string | null>(null);
    const [depProg, setDepProg] = useState<{ processed: number; total: number } | null>(null);
    const [lastDepEdge, setLastDepEdge] = useState<{ source_id: string; target_id: string; ts: number } | null>(null);
    const linksRef = useRef<Record<string, Set<string>>>({});
    const esRef = useRef<EventSource | null>(null);
    const bankRef = useRef<DefinitionBank | null>(null);
    const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";
    const searchParams = useSearchParams();
    const readerRef = useRef<PaperReaderHandle | null>(null);
    const pdfUrl = useMemo(() => getPdfUrl(arxivId), [arxivId]);
    const [anchors, setAnchors] = useState<ArtifactAnchorIndex | null>(null);
    const [exportMenuOpen, setExportMenuOpen] = useState(false);
    const [toast, setToast] = useState<string | null>(null);

    useEffect(() => {
        let cancelled = false;
        if (!arxivId) return;
        getAnchors(arxivId)
            .then((a) => {
                if (!cancelled) setAnchors(a);
            })
            .catch(() => {
                if (!cancelled) setAnchors(null); // 404 or network -> treat as no anchors yet
            });
        return () => {
            cancelled = true;
        };
    }, [arxivId]);

    const artifactLookup = useMemo(() => {
        const g = (liveGraph ?? data?.graph) || null;
        const map: Record<string, string[]> = {};
        if (g) {
            g.nodes.forEach((n) => {
                const arr: string[] = [];
                if (n.display_name) arr.push(n.display_name);
                if (n.label) arr.push(String(n.label));
                arr.push(n.type);
                const uniq = Array.from(new Set(arr.filter(Boolean).map((s) => s.trim())));
                map[n.id] = uniq;
            });
        }
        return map;
    }, [liveGraph, data]);

    useEffect(() => {
        if (streaming) return;
        setLiveGraph((prev) => {
            if (!data?.graph) return null;
            // Prevent adopting a snapshot with fewer edges than current live graph
            if (prev && (data.graph.edges?.length ?? 0) < (prev.edges?.length ?? 0)) return prev;
            return data.graph;
        });
        setLiveBank(data?.definition_bank || null);
        bankRef.current = data?.definition_bank || null;
        linksRef.current = {};
    }, [data, streaming]);

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
        setLastDepEdge(null);

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
                                const bankObj = bank as any;
                                // 1) Direct key lookup
                                let def = bankObj ? bankObj[t] : null;

                                // 2) Fallback: find by exact term (case-insensitive)
                                if (!def && bankObj) {
                                    const values = Object.values(bankObj) as any[];
                                    const norm = (s: string) => (s || "").toLowerCase().trim();
                                    def =
                                        values.find((e: any) => norm(e?.term) === norm(t)) ||
                                        // 3) Fallback: find by alias match
                                        values.find(
                                            (e: any) => Array.isArray(e?.aliases) && e.aliases.some((a: string) => norm(a) === norm(t))
                                        ) ||
                                        null;
                                }

                                const text = def?.definition_text || "";
                                // Keep term visible and show its resolved definition text (if any)
                                return `<div style="margin-bottom:6px"><strong>${t}:</strong> ${text}</div>`;
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
                setLastDepEdge({ source_id: p.source_id, target_id: p.target_id, ts: Date.now() });
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
    const selectedText = useMemo(() => {
        if (!selected) return null as string | null;
        try {
            const html = sanitizePreview(unescapeLatex(selected.content || ""));
            const text = html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
            return text || null;
        } catch {
            return null;
        }
    }, [selected]);
    const [graphModalOpen, setGraphModalOpen] = useState(false);
    const [graphZoom, setGraphZoom] = useState(1);
    const [defsOpen, setDefsOpen] = useState(true);

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
                <main className="min-h-screen p-6 md:p-12 bg-slate-50">
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
                <main className="min-h-screen p-6 md:p-12 bg-slate-50">
                    <div className="max-w-5xl mx-auto space-y-4">
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
                                <span className="text-xs text-slate-600">
                                    Stage: {stage}
                                    {stage === "dependency_inference" && lastDepEdge
                                        ? ` · Inferring: ${(artifactLookup[lastDepEdge.source_id]?.[0] || lastDepEdge.source_id)} -> ${(artifactLookup[lastDepEdge.target_id]?.[0] || lastDepEdge.target_id)}`
                                        : ""}
                                </span>
                            ) : null}
                        </div>
                    </div>
                </main>
            </>
        );
    }

    const { graph, definition_bank } = data;

    function doExportJson(pretty?: boolean) {
        try {
            const g = (liveGraph ?? graph) as DocumentGraph | null;
            if (!g) return;
            const bank = (liveBank ?? definition_bank) as DefinitionBank | null;
            const bundle = buildExportBundle(arxivId, (data?.title ?? null), g, bank);
            const filename = suggestExportFilename(arxivId, "json");
            downloadBundleAsJson(bundle, filename, { pretty: pretty ?? true });
            setToast(`Exported ${filename}`);
            window.setTimeout(() => setToast(null), 2000);
        } catch (e) {
            setToast(`Export failed`);
            window.setTimeout(() => setToast(null), 3000);
        }
    }

    async function doExportGzip(pretty?: boolean) {
        try {
            const g = (liveGraph ?? graph) as DocumentGraph | null;
            if (!g) return;
            const bank = (liveBank ?? definition_bank) as DefinitionBank | null;
            const bundle = buildExportBundle(arxivId, (data?.title ?? null), g, bank);
            const filename = suggestExportFilename(arxivId, "json.gz");
            await downloadBundleAsGzip(bundle, filename, { pretty: pretty ?? false });
            setToast(`Exported ${filename}`);
            window.setTimeout(() => setToast(null), 2000);
        } catch (e: any) {
            const msg = (e && e.message) ? e.message : "Compression not supported. Try JSON export.";
            setToast(msg);
            window.setTimeout(() => setToast(null), 4000);
        }
    }

    return (
        <>
            <header className="sticky top-0 z-40 bg-[var(--surface)] border-b text-[var(--text)]">
                <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4 justify-start">
                    <Link href="/" className="inline-flex items-center gap-2 leading-none shrink-0">
                        <Logo className="h-12 w-auto" withText={true} />
                    </Link>
                </div>
            </header>
            <main className="min-h-screen p-6 md:p-12 bg-slate-50">
                <div className="max-w-5xl mx-auto">
                    <div className="mb-6">
                        <div className="flex items-start justify-between gap-3">
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
                            <div className="relative">
                                <button
                                    className="inline-flex items-center gap-1.5 rounded bg-blue-600 text-white px-2.5 py-1.5 text-xs hover:bg-blue-700"
                                    onClick={() => setExportMenuOpen((s) => !s)}
                                    aria-label="Download graph and definitions"
                                    aria-haspopup="menu"
                                    aria-expanded={exportMenuOpen}
                                    title="Download graph and definitions"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                        <path d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                    <span>Download</span>
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                        <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                </button>
                                {exportMenuOpen ? (
                                    <div role="menu" className="absolute right-0 mt-2 w-56 rounded-md border bg-white shadow-lg z-50">
                                        <button
                                            role="menuitem"
                                            className="w-full text-left px-3 py-2 hover:bg-gray-50 text-sm"
                                            onClick={() => {
                                                setExportMenuOpen(false);
                                                doExportJson(true);
                                            }}
                                        >
                                            Download as JSON
                                        </button>
                                        <button
                                            role="menuitem"
                                            className="w-full text-left px-3 py-2 hover:bg-gray-50 text-sm"
                                            onClick={() => {
                                                setExportMenuOpen(false);
                                                doExportGzip(false);
                                            }}
                                        >
                                            Download as JSON (compressed)
                                        </button>
                                    </div>
                                ) : null}
                            </div>
                        </div>
                        {data.authors && data.authors.length ? (
                            <div className="mt-1 text-sm text-slate-700">{data.authors.join(", ")}</div>
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
                                {stage === "dependency_inference" && lastDepEdge
                                    ? ` · Inferring: ${(artifactLookup[lastDepEdge.source_id]?.[0] || lastDepEdge.source_id)} -> ${(artifactLookup[lastDepEdge.target_id]?.[0] || lastDepEdge.target_id)}`
                                    : ""}
                            </span>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-3 md:p-5">
                        <PanelGroup direction="horizontal">
                            <Panel defaultSize={66} minSize={35}>
                                <div className="flex flex-col gap-2 h-full">
                                    <div className="bg-white rounded-xl ring-1 ring-slate-200 p-2 md:p-3 shadow-sm">
                                        <PaperReader
                                            ref={readerRef}
                                            arxivId={arxivId}
                                            pdfUrl={pdfUrl}
                                            selectedArtifactId={selected?.id || null}
                                            selectedArtifactText={selectedText}
                                            artifactLookup={artifactLookup}
                                            anchors={anchors ?? undefined}
                                            onArtifactClick={(id) => {
                                                const g = liveGraph ?? graph;
                                                const node = g?.nodes.find((n) => n.id === id) || null;
                                                if (node) setSelected(node);
                                            }}
                                        />
                                    </div>
                                    <div className="bg-white rounded-xl ring-1 ring-slate-200 p-2 md:p-3 shadow-sm">
                                        <div className="flex items-center justify-between">
                                            <div className="text-xs font-medium text-slate-600">Definition Bank</div>
                                            <button
                                                className="inline-flex items-center justify-center rounded bg-gray-100 hover:bg-gray-200 p-1.5"
                                                onClick={() => setDefsOpen((o) => !o)}
                                                aria-label={defsOpen ? "Hide definition bank" : "Show definition bank"}
                                                title={defsOpen ? "Hide definition bank" : "Show definition bank"}
                                            >
                                                {defsOpen ? (
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                                        <path d="M6 15l6-6 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                                    </svg>
                                                ) : (
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                                        <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                                    </svg>
                                                )}
                                            </button>
                                        </div>
                                        {defsOpen ? (
                                            <div className="mt-2 overflow-auto max-h-[28vh]">
                                                <DefinitionBankView bank={liveBank ?? definition_bank} />
                                            </div>
                                        ) : null}
                                    </div>
                                </div>
                            </Panel>
                            <PanelResizeHandle className="w-1 bg-slate-200 hover:bg-slate-300 transition-colors" />
                            <Panel defaultSize={34} minSize={25}>
                                <div className="flex flex-col gap-2 h-full">
                                    <div className="bg-white rounded-xl ring-1 ring-slate-200 p-2 md:p-3 shadow-sm">
                                        <div className="flex items-center justify-between mb-1">
                                            <div className="text-xs font-medium text-slate-600">Graph</div>
                                            <div className="inline-flex items-center gap-1.5">
                                                <button
                                                    className="inline-flex items-center justify-center rounded bg-gray-100 hover:bg-gray-200 p-1.5"
                                                    onClick={() => setGraphZoom((z) => Math.max(0.6, Math.round((z - 0.1) * 10) / 10))}
                                                    aria-label="Zoom out"
                                                    title="Zoom out"
                                                >
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                                        <path d="M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                                    </svg>
                                                </button>
                                                <button
                                                    className="inline-flex items-center justify-center rounded bg-gray-100 hover:bg-gray-200 p-1.5"
                                                    onClick={() => setGraphZoom((z) => Math.min(2.0, Math.round((z + 0.1) * 10) / 10))}
                                                    aria-label="Zoom in"
                                                    title="Zoom in"
                                                >
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                                        <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                                    </svg>
                                                </button>
                                                <button
                                                    className="inline-flex items-center justify-center rounded bg-gray-100 hover:bg-gray-200 p-1.5"
                                                    onClick={() => setGraphModalOpen(true)}
                                                    aria-label="Open fullscreen graph"
                                                    title="Open fullscreen graph"
                                                >
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                                        <path d="M7 3H3v4M17 3h4v4M7 21H3v-4M21 17v4h-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                                    </svg>
                                                </button>
                                                <button
                                                    className="inline-flex items-center justify-center rounded bg-gray-100 hover:bg-gray-200 p-1.5"
                                                    onClick={() => doExportJson(true)}
                                                    aria-label="Download graph and definitions"
                                                    title="Download graph and definitions"
                                                >
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                                        <path d="M12 3v10m0 0l-4-4m4 4l4-4M5 21h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                                    </svg>
                                                </button>
                                            </div>
                                        </div>
                                        <D3GraphView
                                            graph={liveGraph ?? graph}
                                            onSelectNode={(node) => {
                                                if (!node) return;
                                                setSelected(node);
                                                try {
                                                    readerRef.current?.scrollToArtifact(node.id);
                                                    readerRef.current?.pulseArtifact(node.id);
                                                } catch { }
                                            }}
                                            selectedNodeId={selected?.id || null}
                                            zoom={graphZoom}
                                            height="42vh"
                                        />
                                    </div>
                                    <div className="bg-white rounded-xl ring-1 ring-slate-200 p-3 shadow-sm h-full overflow-auto">
                                        <div className="text-sm font-semibold mb-2 text-[var(--text)]">Artifact Details</div>
                                        <ArtifactDetails node={selected} bank={liveBank ?? definition_bank} />
                                    </div>
                                </div>
                            </Panel>
                        </PanelGroup>
                    </div>
                </div>

                {toast ? (
                    <div className="fixed bottom-4 right-4 z-50 rounded bg-gray-900 text-white text-sm px-3 py-2 shadow-lg">
                        {toast}
                    </div>
                ) : null}

                {graphModalOpen ? (
                    <div className="fixed inset-0 z-50 flex items-center justify-center">
                        <div
                            className="absolute inset-0 bg-black/50"
                            aria-hidden
                            onClick={() => setGraphModalOpen(false)}
                        />
                        <div className="relative bg-white rounded-xl shadow-xl ring-1 ring-slate-200 w-[min(96vw,1100px)] max-w-[1100px]">
                            <div className="flex items-center justify-between p-2 border-b">
                                <div className="text-xs font-medium text-slate-600 px-1">Graph</div>
                                <button
                                    className="inline-flex items-center justify-center rounded bg-gray-100 hover:bg-gray-200 p-1.5"
                                    onClick={() => setGraphModalOpen(false)}
                                    aria-label="Close graph"
                                    title="Close"
                                >
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                        <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                </button>
                            </div>
                            <div className="p-2">
                                <D3GraphView
                                    graph={liveGraph ?? graph}
                                    onSelectNode={(node) => {
                                        if (!node) return;
                                        setSelected(node);
                                        try {
                                            readerRef.current?.scrollToArtifact(node.id);
                                            readerRef.current?.pulseArtifact(node.id);
                                        } catch { }
                                    }}
                                    selectedNodeId={selected?.id || null}
                                    zoom={graphZoom}
                                    height="78vh"
                                />
                            </div>
                        </div>
                    </div>
                ) : null}

            </main>
        </>
    );
}
