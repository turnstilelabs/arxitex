'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useSearchParams } from 'next/navigation';

import Image from 'next/image';

import Graph, { type ConstellationsGraphHandle } from '@/components/Graph';
import GraphFeedbackModal from '@/components/GraphFeedbackModal';

const BACKEND_URL = process.env.NEXT_PUBLIC_ARXITEX_BACKEND_URL ?? 'http://127.0.0.1:8000';

type ProcessStats = {
    artifacts: number;
    links: number;
};

function normalizeEdgeForDisplay(edge: any): {
    source: string;
    target: string;
    dependency_type: string;
    reference_type?: string | null;
} | null {
    const rawS = String(edge?.source ?? edge?.source_id ?? edge?.sourceId ?? '');
    const rawT = String(edge?.target ?? edge?.target_id ?? edge?.targetId ?? '');

    // Mirror the frontend normalization in `processGraphData`.
    const depRaw = String(edge?.dependency_type ?? edge?.dependencyType ?? 'internal');

    if (depRaw === 'provides_remark') {
        // Dropped in `processGraphData`.
        return null;
    }

    const ref = (edge?.reference_type ?? edge?.referenceType ?? null) as string | null;

    // Reverse direction for “used_in” semantics
    if (depRaw === 'uses_result' || depRaw === 'uses_definition' || depRaw === 'is_corollary_of') {
        return {
            source: rawT,
            target: rawS,
            dependency_type: 'used_in',
            reference_type: ref,
        };
    }

    // Reverse direction for “generalized_by” semantics
    if (depRaw === 'is_generalization_of') {
        return {
            source: rawT,
            target: rawS,
            dependency_type: 'generalized_by',
            reference_type: ref,
        };
    }

    return {
        source: rawS,
        target: rawT,
        dependency_type: depRaw || 'internal',
        reference_type: ref,
    };
}

function getDisplayedEdgeKey(edge: any): string {
    const n = normalizeEdgeForDisplay(edge);
    if (!n) return '';

    // Mirror how the graph dedupes/join keys (see `applyMutations` in incremental.ts)
    return `${n.source}=>${n.target}::${n.dependency_type || 'internal'}`;
}


type PaperMeta = {
    title: string;
    authors: string[];
};

async function fetchArxivMetadata(arxivId: string): Promise<PaperMeta> {
    const res = await fetch(`/api/arxiv-metadata?arxivId=${encodeURIComponent(arxivId)}`, {
        cache: 'no-store',
    });

    if (!res.ok) {
        throw new Error(`Failed to fetch paper metadata (status ${res.status})`);
    }

    const json = (await res.json()) as { title: string; authors: string[] };
    return { title: json.title, authors: json.authors };
}

export default function PaperPage() {
    const params = useParams<{ arxivId: string }>();
    const searchParams = useSearchParams();

    const arxivId = params.arxivId;

    const inferDependencies = useMemo(() => {
        const v = searchParams.get('deps');
        return v == null ? true : v !== '0';
    }, [searchParams]);

    const enrichContent = useMemo(() => {
        const v = searchParams.get('enrich');
        return v == null ? true : v !== '0';
    }, [searchParams]);

    const [paperMeta, setPaperMeta] = useState<PaperMeta | null>(null);
    const [paperMetaError, setPaperMetaError] = useState<string | null>(null);

    const graphRef = useRef<ConstellationsGraphHandle | null>(null);
    const [status, setStatus] = useState<string[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [stats, setStats] = useState<ProcessStats>({
        artifacts: 0,
        links: 0,
    });

    const [feedbackOpen, setFeedbackOpen] = useState(false);
    const [feedbackScope, setFeedbackScope] = useState<'graph' | 'node'>('graph');
    const [feedbackNodeId, setFeedbackNodeId] = useState<string | null>(null);
    const [feedbackContextLabel, setFeedbackContextLabel] = useState<string | undefined>(undefined);

    const statsRef = useRef({
        nodeIds: new Set<string>(),
        edgeKeys: new Set<string>(),
    });

    const absUrl = `https://arxiv.org/abs/${arxivId}`;

    useEffect(() => {
        let cancelled = false;
        setPaperMeta(null);
        setPaperMetaError(null);

        fetchArxivMetadata(arxivId)
            .then((m) => {
                if (cancelled) return;
                setPaperMeta(m);
            })
            .catch((e: any) => {
                if (cancelled) return;
                setPaperMetaError(e?.message ?? String(e));
            });

        return () => {
            cancelled = true;
        };
    }, [arxivId]);

    useEffect(() => {
        let cancelled = false;

        async function run() {
            setIsLoading(true);
            setError(null);
            graphRef.current?.reset();

            // Reset process stats for this run.
            statsRef.current.nodeIds = new Set();
            statsRef.current.edgeKeys = new Set();
            setStats({ artifacts: 0, links: 0 });

            setStatus(['Initiating request...']);

            try {
                const response = await fetch(`${BACKEND_URL}/process-paper`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ arxivUrl: absUrl, enrichContent, inferDependencies }),
                });

                if (!response.ok || !response.body) {
                    throw new Error(`Request failed with status: ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (!cancelled) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    const parts = buffer.split('\n\n');

                    for (let i = 0; i < parts.length - 1; i++) {
                        const part = parts[i];
                        if (part.startsWith('data: ')) {
                            try {
                                const json = JSON.parse(part.substring(6));
                                if (json.type === 'status') {
                                    setStatus((prev) => [...prev, json.data]);
                                } else if (json.type === 'node') {
                                    const nodeId = String(json?.data?.id ?? '');
                                    if (nodeId && !statsRef.current.nodeIds.has(nodeId)) {
                                        statsRef.current.nodeIds.add(nodeId);
                                        setStats((prev) => ({
                                            ...prev,
                                            artifacts: statsRef.current.nodeIds.size,
                                        }));
                                    }

                                    graphRef.current?.ingest({ type: 'node', data: json.data });
                                } else if (json.type === 'link') {
                                    // Count edges the same way the graph renderer will display them.
                                    const k = getDisplayedEdgeKey(json.data);
                                    if (k && !statsRef.current.edgeKeys.has(k)) {
                                        statsRef.current.edgeKeys.add(k);

                                        setStats((prev) => ({
                                            ...prev,
                                            links: statsRef.current.edgeKeys.size,
                                        }));
                                    }

                                    graphRef.current?.ingest({ type: 'link', data: json.data });
                                } else if (json.type === 'done') {
                                    // no-op
                                }
                            } catch {
                                console.error('Failed to parse stream chunk:', part);
                            }
                        }
                    }
                    buffer = parts[parts.length - 1];
                }
            } catch (err: any) {
                if (cancelled) return;
                setError(err.message);
                setStatus((prev) => [...prev, `Error: ${err.message}`]);
            } finally {
                if (cancelled) return;
                setIsLoading(false);
            }
        }

        run();

        return () => {
            cancelled = true;
        };
    }, [absUrl, enrichContent, inferDependencies]);

    return (
        <main
            className="flex flex-col items-center min-h-screen p-4 sm:p-8 md:p-12"
            style={{ background: 'var(--background)', color: 'var(--primary-text)' }}
        >
            <div className="w-full max-w-6xl">
                <div
                    style={{
                        background: 'var(--surface1)',
                        border: '1px solid var(--border-color)',
                        borderRadius: 12,
                        padding: 16,
                    }}
                >
                    <div className="flex items-start justify-between gap-4 flex-wrap">
                        <div>
                            <div className="flex items-center gap-3 flex-wrap">
                                <h1
                                    className="text-2xl sm:text-3xl font-black tracking-tight"
                                    style={{ color: 'var(--accent)' }}
                                >
                                    {paperMeta?.title ?? 'Loading paper…'}
                                </h1>
                                <a
                                    href={absUrl}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="paper-link-btn"
                                    aria-label="Open arXiv abstract"
                                    title="Open arXiv abstract"
                                >
                                    <Image src="/globe.svg" alt="arXiv" width={18} height={18} />
                                </a>
                            </div>

                            {paperMeta?.authors?.length ? (
                                <div
                                    className="mt-2 text-sm"
                                    style={{ color: 'var(--secondary-text)' }}
                                >
                                    {paperMeta.authors.join(', ')}
                                </div>
                            ) : paperMetaError ? (
                                <div
                                    className="mt-2 text-sm"
                                    style={{ color: '#ff6b6b' }}
                                >
                                    Metadata error: {paperMetaError}
                                </div>
                            ) : null}

                            {/* Stats line with inline suggestion flag */}
                            <div
                                className="mt-3 text-sm flex items-center gap-2 flex-wrap"
                                style={{
                                    color: 'var(--secondary-text)',
                                    fontFamily: 'Inter, system-ui, sans-serif',
                                }}
                            >
                                <span>Extracted </span>
                                <strong
                                    style={{ color: 'var(--primary-text)', fontWeight: 700 }}
                                >
                                    {stats.artifacts}
                                </strong>
                                <span> artifacts with </span>
                                <strong
                                    style={{ color: 'var(--primary-text)', fontWeight: 700 }}
                                >
                                    {stats.links}
                                </strong>
                                <span> links</span>
                                {isLoading ? <span> · processing…</span> : null}
                                {error ? (
                                    <span style={{ color: '#ff6b6b' }}> · error</span>
                                ) : null}

                                {/* Suggest a correction button after stats */}
                                <button
                                    type="button"
                                    className="inline-flex items-center justify-center p-0.5 rounded hover:bg-transparent"
                                    style={{ color: 'var(--secondary-text)' }}
                                    aria-label="Suggest a correction for this graph"
                                    title="Suggest a correction"
                                    onClick={() => {
                                        setFeedbackScope('graph');
                                        setFeedbackNodeId(null);
                                        setFeedbackContextLabel(undefined);
                                        setFeedbackOpen(true);
                                    }}
                                >
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        width="14"
                                        height="14"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    >
                                        <path d="M4 22V4" />
                                        <path d="M4 4h12l-1.5 4L20 12H4" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* spacer */}
                </div>

                {error && (
                    <p
                        className="mt-4 font-semibold"
                        style={{ color: '#ff6b6b' }}
                    >
                        Error: {error}
                    </p>
                )}

                <GraphFeedbackModal
                    open={feedbackOpen}
                    onClose={() => setFeedbackOpen(false)}
                    paperId={arxivId}
                    scope={feedbackScope}
                    nodeId={
                        feedbackScope === 'node'
                            ? feedbackNodeId ?? undefined
                            : undefined
                    }
                    contextLabel={feedbackContextLabel}
                />

                <div className="w-full mt-6">
                    <div
                        className="relative w-full h-[70vh] rounded-lg shadow-inner"
                        style={{
                            border: '1px solid var(--border-color)',
                            background: 'var(--background)',
                        }}
                    >
                        <Graph
                            ref={graphRef}
                            onReportNode={(n) => {
                                setFeedbackScope('node');
                                setFeedbackNodeId(n.id);
                                setFeedbackContextLabel(n.label);
                                setFeedbackOpen(true);
                            }}
                        />
                    </div>
                </div>

                <div className="w-full mt-4">
                    <h2 className="font-semibold">Analysis Log:</h2>
                    <div
                        className="mt-2 text-sm h-28 overflow-y-auto p-2 rounded-md"
                        style={{
                            background: 'var(--surface1)',
                            border: '1px solid var(--border-color)',
                            color: 'var(--secondary-text)',
                        }}
                    >
                        {status.map((s, i) => (
                            <p key={i}>{s}</p>
                        ))}
                    </div>
                </div>
            </div>
        </main>
    );
}
