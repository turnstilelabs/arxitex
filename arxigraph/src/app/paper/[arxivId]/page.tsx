'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useSearchParams, useRouter } from 'next/navigation';

import Image from 'next/image';

import Graph, { type ConstellationsGraphHandle } from '@/components/Graph';
import GraphFeedbackModal from '@/components/GraphFeedbackModal';
import ProcessingErrorModal, { type ProcessingError } from '@/components/ProcessingErrorModal';
import DefinitionBankCard, { type DefinitionBankEntry } from '@/components/DefinitionBankCard';
import { typesetMath } from '@/components/constellations/mathjax';

const BACKEND_URL = process.env.NEXT_PUBLIC_ARXITEX_BACKEND_URL ?? 'http://127.0.0.1:8000';

const HF_DATASET_ORG = process.env.NEXT_PUBLIC_HF_DATASET_ORG;
const HF_DATASET_REPO = process.env.NEXT_PUBLIC_HF_DATASET_REPO;
const HF_DATASET_REF = process.env.NEXT_PUBLIC_HF_DATASET_REF ?? 'main';

function hfJsonUrlForArxivId(arxivId: string): string | null {
    if (!HF_DATASET_ORG || !HF_DATASET_REPO) return null;

    const safeId = arxivId.replace('/', '_');
    return `https://huggingface.co/datasets/${HF_DATASET_ORG}/${HF_DATASET_REPO}/resolve/${HF_DATASET_REF}/data/arxiv_${safeId}.json`;
}

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
    const typ = (edge?.type ?? null) as string | null;

    // Canonical semantics (must mirror processGraphData):
    // - For "used_in" edges, arrow points prerequisite -> result.
    // Normalize both raw and already-normalized dependency types.
    if (
        depRaw === 'uses_result' ||
        depRaw === 'uses_definition' ||
        depRaw === 'is_corollary_of' ||
        depRaw === 'used_in'
    ) {
        return {
            source: rawT,
            target: rawS,
            dependency_type: 'used_in',
            reference_type: ref,
        };
    }

    if (depRaw === 'is_generalization_of' || depRaw === 'generalized_by') {
        return {
            source: rawT,
            target: rawS,
            dependency_type: 'generalized_by',
            reference_type: ref,
        };
    }

    // Internal cross-references: referenced node is a prerequisite for the referrer.
    if ((depRaw === 'internal' || !depRaw) && (ref === 'internal' || typ === 'internal')) {
        return {
            source: rawT,
            target: rawS,
            dependency_type: 'internal',
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
    abstract: string;
};

async function fetchArxivMetadata(arxivId: string): Promise<PaperMeta> {
    const res = await fetch(`/api/arxiv-metadata?arxivId=${encodeURIComponent(arxivId)}`, {
        cache: 'no-store',
    });

    if (!res.ok) {
        throw new Error(`Failed to fetch paper metadata (status ${res.status})`);
    }

    const json = (await res.json()) as { title: string; authors: string[]; abstract?: string };
    return { title: json.title, authors: json.authors, abstract: json.abstract ?? '' };
}

export default function PaperPage() {
    const params = useParams<{ arxivId: string }>();
    const searchParams = useSearchParams();
    const router = useRouter();

    const arxivId = params.arxivId;

    const inferDependencies = useMemo(() => {
        const v = searchParams.get('deps');
        return v == null ? true : v !== '0';
    }, [searchParams]);

    const enrichContent = useMemo(() => {
        const v = searchParams.get('enrich');
        return v == null ? true : v !== '0';
    }, [searchParams]);

    // Fallback label based on which analyses are enabled, used when we
    // don't yet have (or can't parse) a more specific stage label from
    // the backend streaming status events.
    const analysisLabel = useMemo(() => {
        if (inferDependencies && enrichContent) {
            return 'processing paper';
        }
        if (inferDependencies) {
            return 'inferring dependencies between artifacts';
        }
        if (enrichContent) {
            return 'enriching symbols and definitions for artifacts';
        }
        return 'building base graph';
    }, [inferDependencies, enrichContent]);

    const hasFullAnalysis = enrichContent && inferDependencies;

    const [paperMeta, setPaperMeta] = useState<PaperMeta | null>(null);
    const [paperMetaError, setPaperMetaError] = useState<string | null>(null);

    const abstractRef = useRef<HTMLDivElement | null>(null);

    function normalizeAbstract(s: string) {
        // Keep in sync with other MathJax-enabled renderers in the app.
        // - De-double-escape backslashes ("\\\\alpha" -> "\\alpha")
        // - Strip LaTeX labels that can confuse MathJax
        return String(s)
            .replace(/\\\\/g, '\\')
            .replace(/\\label\{[^}]*\}/g, '')
            .trim();
    }

    const graphRef = useRef<ConstellationsGraphHandle | null>(null);
    const [status, setStatus] = useState<string[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [processingError, setProcessingError] = useState<ProcessingError | null>(null);
    const [isErrorModalOpen, setIsErrorModalOpen] = useState(false);

    const [stats, setStats] = useState<ProcessStats>({
        artifacts: 0,
        links: 0,
    });

    const [feedbackOpen, setFeedbackOpen] = useState(false);
    const [feedbackScope, setFeedbackScope] = useState<'graph' | 'node'>('graph');
    const [feedbackNodeId, setFeedbackNodeId] = useState<string | null>(null);
    const [feedbackContextLabel, setFeedbackContextLabel] = useState<string | undefined>(undefined);
    const [advancedAnalysisOpen, setAdvancedAnalysisOpen] = useState(false);
    const [selectedEnrich, setSelectedEnrich] = useState(false);
    const [selectedDeps, setSelectedDeps] = useState(false);

    const [definitionBank, setDefinitionBank] = useState<DefinitionBankEntry[]>([]);
    const latexMacrosRef = useRef<Record<string, string>>({});

    // Track the current high-level pipeline stage based on streamed
    // backend status messages (base graph, enrichment, dependencies).
    const [stageLabel, setStageLabel] = useState<string | null>(null);

    const statsRef = useRef({
        nodeIds: new Set<string>(),
        edgeKeys: new Set<string>(),
    });

    // Ensure we only reset the graph once per run, and only when the first
    // node of the new run arrives. This avoids mixing graphs across runs
    // while still keeping the previous graph visible while the backend works.
    const shouldResetGraphRef = useRef(false);

    const absUrl = `https://arxiv.org/abs/${arxivId}`;

    const runAdvancedAnalysis = (nextEnrich: boolean, nextDeps: boolean) => {
        const params = new URLSearchParams(searchParams.toString());
        params.set('enrich', nextEnrich ? '1' : '0');
        params.set('deps', nextDeps ? '1' : '0');

        setAdvancedAnalysisOpen(false);
        router.replace(`/paper/${encodeURIComponent(arxivId)}?${params.toString()}`);
    };

    useEffect(() => {
        if (!advancedAnalysisOpen) return;
        // Pre-select any analysis that hasn't been run yet.
        setSelectedEnrich(!enrichContent);
        setSelectedDeps(!inferDependencies);
    }, [advancedAnalysisOpen, enrichContent, inferDependencies]);

    const canRunSelected = (!enrichContent && selectedEnrich) || (!inferDependencies && selectedDeps);

    const handleRunSelectedAnalysis = () => {
        if (!canRunSelected || isLoading) return;

        const nextEnrich = enrichContent || selectedEnrich;
        const nextDeps = inferDependencies || selectedDeps;

        runAdvancedAnalysis(nextEnrich, nextDeps);
    };

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
        if (!paperMeta?.abstract) return;
        // MathJax is loaded globally in `src/app/layout.tsx`, but we need to
        // manually typeset dynamic client-rendered content.
        void typesetMath([abstractRef.current]);
    }, [paperMeta?.abstract]);

    useEffect(() => {
        let cancelled = false;

        async function runViaBackend() {
            console.log('[Backend] Falling back to backend /process-paper for', arxivId);

            setIsLoading(true);
            setError(null);
            setProcessingError(null);
            setIsErrorModalOpen(false);
            setStageLabel(null);

            // Reset process stats for this run.
            statsRef.current.nodeIds = new Set();
            statsRef.current.edgeKeys = new Set();
            setStats({ artifacts: 0, links: 0 });

            // Mark that the next node we see belongs to a fresh run.
            shouldResetGraphRef.current = true;

            setStatus((prev) => [
                ...prev,
                'Initiating backend request... (HF dataset not available or fetch failed)',
            ]);

            try {
                console.log('[Backend] POST', `${BACKEND_URL}/process-paper`, {
                    arxivUrl: absUrl,
                    enrichContent,
                    inferDependencies,
                });

                const response = await fetch(`${BACKEND_URL}/process-paper`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ arxivUrl: absUrl, enrichContent, inferDependencies }),
                });

                console.log('[Backend] Response status', response.status);

                // Handle early failures (e.g. validation errors, missing API keys)
                if (!response.ok) {
                    let details: any = null;
                    try {
                        details = await response.json();
                    } catch {
                        // ignore JSON parse errors; we'll fall back to a generic message.
                    }

                    if (details) {
                        const rawDetail = details.detail as string | undefined;

                        // Map common FastAPI 400 detail messages into our
                        // structured error codes so the UI can show tailored copy.
                        let inferredCode: string | undefined;
                        if (typeof rawDetail === 'string') {
                            const lower = rawDetail.toLowerCase();
                            if (lower.includes('openai_api_key is not set')) {
                                inferredCode = 'enhancements_misconfigured';
                            } else if (
                                lower.includes('could not extract arxiv id') ||
                                lower.includes('invalid arxiv id')
                            ) {
                                inferredCode = 'invalid_arxiv_id';
                            }
                        }

                        const code: string =
                            details.reason_code ??
                            details.code ??
                            inferredCode ??
                            'unexpected_error';

                        const message: string =
                            details.reason ??
                            details.message ??
                            rawDetail ??
                            `Request failed with status: ${response.status}`;

                        const stage = details.error_stage ?? details.stage;

                        const err: ProcessingError = { code, message, stage };
                        setProcessingError(err);
                        setIsErrorModalOpen(true);
                        setError(message);
                        setStatus((prev) => [...prev, `Error: ${message}`]);
                    } else {
                        const message = `Request failed with status: ${response.status}`;
                        setError(message);
                        setStatus((prev) => [...prev, `Error: ${message}`]);
                    }

                    return;
                }

                if (!response.body) {
                    const message = 'Server did not return a response body.';
                    setError(message);
                    setStatus((prev) => [...prev, `Error: ${message}`]);
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                function parseSseEvents(chunk: string): Array<{ type: string; data?: any }> {
                    // Split on blank lines (handle \n\n and \r\n\r\n).
                    const blocks = chunk.split(/\r?\n\r?\n/);
                    const completeBlocks = blocks.slice(0, -1);
                    buffer = blocks[blocks.length - 1];

                    const events: Array<{ type: string; data?: any }> = [];
                    for (const block of completeBlocks) {
                        const lines = block.split(/\r?\n/);
                        const dataLines: string[] = [];
                        for (const line of lines) {
                            if (!line) continue;
                            if (line.startsWith(':')) continue; // comment/keepalive
                            if (line.startsWith('data:')) {
                                dataLines.push(line.slice(5).trimStart());
                            }
                        }
                        if (!dataLines.length) continue;
                        const payload = dataLines.join('\n');
                        try {
                            events.push(JSON.parse(payload));
                        } catch {
                            console.error('Failed to parse stream chunk:', `data: ${payload}`);
                        }
                    }
                    return events;
                }

                while (!cancelled) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const events = parseSseEvents(buffer);

                    for (const json of events) {
                        if (json.type === 'status') {
                            // Drop keep-alives from log.
                            if (json.data !== 'keep-alive') {
                                setStatus((prev) => [...prev, json.data]);

                                const msg = String(json.data ?? '').toLowerCase();
                                if (msg.includes('building base graph')) {
                                    setStageLabel('building base graph');
                                } else if (msg.includes('enriching symbols and definitions')) {
                                    setStageLabel('enriching symbols and definitions for artifacts');
                                } else if (msg.includes('inferring dependencies between artifacts')) {
                                    setStageLabel('inferring dependencies between artifacts');
                                } else if (msg.includes('graph extraction complete')) {
                                    // Keep the last meaningful stage; the inline
                                    // label will disappear once isLoading=false.
                                }
                            }
                        } else if (json.type === 'latex_macros') {
                            const m = (json.data ?? {}) as Record<string, string>;
                            latexMacrosRef.current = m;

                            // Merge per-paper macros into the global MathJax config.
                            const win = window as any;
                            if (win.MathJax && win.MathJax.config && win.MathJax.config.tex) {
                                win.MathJax.config.tex.macros = {
                                    ...(win.MathJax.config.tex.macros || {}),
                                    ...m,
                                };
                            } else if (win.MathJax && win.MathJax.tex) {
                                win.MathJax.tex.macros = {
                                    ...(win.MathJax.tex.macros || {}),
                                    ...m,
                                };
                            }
                        } else if (json.type === 'node') {
                            if (shouldResetGraphRef.current) {
                                graphRef.current?.reset();
                                shouldResetGraphRef.current = false;
                            }

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
                            const k = getDisplayedEdgeKey(json.data);
                            if (k && !statsRef.current.edgeKeys.has(k)) {
                                statsRef.current.edgeKeys.add(k);
                                setStats((prev) => ({
                                    ...prev,
                                    links: statsRef.current.edgeKeys.size,
                                }));
                            }

                            graphRef.current?.ingest({ type: 'link', data: json.data });
                        } else if (json.type === 'definition_bank') {
                            const raw = json.data ?? {};
                            const entries: DefinitionBankEntry[] = Object.values(raw).map(
                                (d: any) => ({
                                    term: String(d?.term ?? ''),
                                    definitionText: String(d?.definition_text ?? ''),
                                    aliases: Array.isArray(d?.aliases) ? d.aliases : [],
                                }),
                            );

                            entries.sort((a, b) =>
                                a.term.localeCompare(b.term, undefined, { sensitivity: 'base' }),
                            );

                            setDefinitionBank(entries);
                        } else if (json.type === 'error') {
                            const details = json.data ?? {};
                            const code: string =
                                details.reason_code ??
                                details.code ??
                                'unexpected_error';
                            const message: string =
                                details.reason ??
                                details.message ??
                                'An unexpected error occurred while processing this paper.';
                            const stage = details.error_stage ?? details.stage;

                            const err: ProcessingError = { code, message, stage };
                            setProcessingError(err);
                            setIsErrorModalOpen(true);
                            setError(message);
                            setStatus((prev) => [...prev, `Error: ${message}`]);

                            cancelled = true;
                            try {
                                void reader.cancel();
                            } catch {
                                // ignore
                            }
                            break;
                        }
                    }
                }
            } catch (err: any) {
                if (cancelled) return;

                const message: string =
                    (err && typeof err.message === 'string')
                        ? err.message
                        : 'Failed to contact processing backend.';

                // Treat network/transport failures as a structured backend error so
                // users see the dedicated modal instead of a raw "Failed to fetch".
                const backendError: ProcessingError = {
                    code: 'backend_unreachable',
                    message,
                    stage: 'backend',
                };

                setProcessingError(backendError);
                setIsErrorModalOpen(true);
                setError(message);
                setStatus((prev) => [...prev, `Error: ${message}`]);
            } finally {
                if (cancelled) return;
                setIsLoading(false);
            }
        }

        async function run() {
            setIsLoading(true);
            setError(null);
            setProcessingError(null);
            setIsErrorModalOpen(false);
            setStageLabel(null);

            // Reset process stats for this run.
            statsRef.current.nodeIds = new Set();
            statsRef.current.edgeKeys = new Set();
            setStats({ artifacts: 0, links: 0 });

            // Mark that the next node we see belongs to a fresh run.
            shouldResetGraphRef.current = true;

            setStatus(['Initiating request...']);

            const hfUrl = hfJsonUrlForArxivId(arxivId);

            if (hfUrl) {
                console.log('[HF] Trying Hugging Face URL', hfUrl, 'for', arxivId);
                setStatus((prev) => [...prev, `Trying Hugging Face dataset at ${hfUrl}`]);
                try {
                    const res = await fetch(hfUrl, {
                        // Data is immutable when HF_DATASET_REF is a commit hash.
                        cache: 'force-cache',
                    });

                    console.log('[HF] Response status', res.status, 'for', hfUrl);

                    if (res.ok) {
                        const payload = await res.json();
                        console.log('[HF] Successfully loaded payload from Hugging Face for', arxivId);
                        const graph = payload?.graph ?? {};
                        const nodes: any[] = Array.isArray(graph?.nodes) ? graph.nodes : [];
                        const edges: any[] = Array.isArray(graph?.edges) ? graph.edges : [];

                        // Reset and ingest full snapshot.
                        graphRef.current?.reset();
                        statsRef.current.nodeIds = new Set();
                        statsRef.current.edgeKeys = new Set();

                        for (const node of nodes) {
                            const nodeId = String(node?.id ?? '');
                            if (nodeId && !statsRef.current.nodeIds.has(nodeId)) {
                                statsRef.current.nodeIds.add(nodeId);
                            }
                            graphRef.current?.ingest({ type: 'node', data: node });
                        }

                        for (const edge of edges) {
                            const k = getDisplayedEdgeKey(edge);
                            if (k && !statsRef.current.edgeKeys.has(k)) {
                                statsRef.current.edgeKeys.add(k);
                            }
                            graphRef.current?.ingest({ type: 'link', data: edge });
                        }

                        setStats({
                            artifacts: statsRef.current.nodeIds.size,
                            links: statsRef.current.edgeKeys.size,
                        });

                        const rawBank = payload?.definition_bank ?? null;
                        if (rawBank && typeof rawBank === 'object') {
                            const entries: DefinitionBankEntry[] = Object.values(rawBank).map((d: any) => ({
                                term: String(d?.term ?? ''),
                                definitionText: String(d?.definition_text ?? ''),
                                aliases: Array.isArray(d?.aliases) ? d.aliases : [],
                            }));

                            entries.sort((a, b) =>
                                a.term.localeCompare(b.term, undefined, { sensitivity: 'base' }),
                            );

                            setDefinitionBank(entries);
                        } else {
                            setDefinitionBank([]);
                        }

                        setStatus((prev) => [...prev, 'Loaded from Hugging Face dataset.']);
                        setIsLoading(false);
                        return;
                    }
                } catch (e: any) {
                    // Log and fall back to backend.
                    console.error('Failed to load from Hugging Face dataset:', e);
                    setStatus((prev) => [
                        ...prev,
                        'Hugging Face fetch failed, falling back to backend.',
                    ]);
                }
            }

            // Fallback: live backend streaming.
            await runViaBackend();
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
                            ) : paperMetaError && !processingError ? (
                                // When we have a structured processing error, prefer the
                                // dedicated modal instead of showing this inline metadata
                                // error line.
                                <div
                                    className="mt-2 text-sm"
                                    style={{ color: '#ff6b6b' }}
                                >
                                    Metadata error: {paperMetaError}
                                </div>
                            ) : null}

                            {paperMeta?.abstract ? (
                                <div
                                    ref={abstractRef}
                                    className="mt-3 text-sm leading-relaxed"
                                    style={{
                                        background: 'var(--surface2)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: 10,
                                        padding: 12,
                                        color: 'var(--primary-text)',
                                    }}
                                >
                                    {normalizeAbstract(paperMeta.abstract)}
                                </div>
                            ) : null}

                            {/* Processing / status line (kept under abstract) */}
                            <div
                                className="mt-3 text-sm flex items-center gap-2 flex-wrap"
                                style={{
                                    color: 'var(--secondary-text)',
                                    fontFamily: 'Inter, system-ui, sans-serif',
                                }}
                            >
                                {isLoading ? (
                                    <span>{(stageLabel ?? analysisLabel)}...</span>
                                ) : null}
                                {error ? (
                                    <span style={{ color: '#ff6b6b' }}>error</span>
                                ) : null}

                                {!hasFullAnalysis && (
                                    <button
                                        type="button"
                                        className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border shadow-sm hover:bg-[var(--surface2)] disabled:opacity-60"
                                        style={{
                                            borderColor: 'var(--border-color)',
                                            color: 'var(--secondary-text)',
                                            background: 'transparent',
                                        }}
                                        aria-label="Run additional analysis"
                                        title="Run Additional Analysis"
                                        onClick={() => setAdvancedAnalysisOpen(true)}
                                        disabled={isLoading}
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
                                            <circle cx="12" cy="12" r="4" />
                                            <path d="M4 12h2" />
                                            <path d="M18 12h2" />
                                            <path d="M12 4v2" />
                                            <path d="M12 18v2" />
                                        </svg>
                                        <span>Run Additional Analysis</span>
                                    </button>
                                )}
                            </div>

                        </div>
                    </div>

                    {/* spacer */}
                </div>

                {advancedAnalysisOpen && !hasFullAnalysis && (
                    <div
                        className="fixed inset-0 z-40 flex items-center justify-center px-4"
                        style={{ background: 'rgba(0,0,0,0.4)' }}
                        onClick={() => setAdvancedAnalysisOpen(false)}
                    >
                        <div
                            className="w-full max-w-sm rounded-lg shadow-lg"
                            style={{
                                background: 'var(--surface1)',
                                border: '1px solid var(--border-color)',
                                color: 'var(--primary-text)',
                            }}
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div
                                className="flex items-center justify-between px-4 py-3 border-b"
                                style={{ borderColor: 'var(--border-color)' }}
                            >
                                <h2 className="text-sm font-semibold">Run Additional Analysis</h2>
                                <button
                                    type="button"
                                    onClick={() => setAdvancedAnalysisOpen(false)}
                                    className="p-1 rounded hover:bg-[var(--surface2)]"
                                    aria-label="Close"
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
                                        <line x1="18" y1="6" x2="6" y2="18" />
                                        <line x1="6" y1="6" x2="18" y2="18" />
                                    </svg>
                                </button>
                            </div>
                            <div className="px-4 py-3 text-xs sm:text-sm space-y-3">
                                {!enrichContent && (
                                    <label className="flex items-start gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            className="mt-1 h-3 w-3 accent-[var(--accent)]"
                                            checked={selectedEnrich}
                                            onChange={(e) => setSelectedEnrich(e.target.checked)}
                                        />
                                        <div>
                                            <div className="text-xs sm:text-sm font-semibold">
                                                Enhance artifacts
                                            </div>
                                            <div
                                                className="text-[0.75rem]"
                                                style={{ color: 'var(--secondary-text)' }}
                                            >
                                                Add enriched definitions and symbols.
                                            </div>
                                        </div>
                                    </label>
                                )}
                                {!inferDependencies && (
                                    <label className="flex items-start gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            className="mt-1 h-3 w-3 accent-[var(--accent)]"
                                            checked={selectedDeps}
                                            onChange={(e) => setSelectedDeps(e.target.checked)}
                                        />
                                        <div>
                                            <div className="text-xs sm:text-sm font-semibold">
                                                Infer dependencies
                                            </div>
                                            <div
                                                className="text-[0.75rem]"
                                                style={{ color: 'var(--secondary-text)' }}
                                            >
                                                Add inferred result/definition links.
                                            </div>
                                        </div>
                                    </label>
                                )}
                                <button
                                    type="button"
                                    onClick={handleRunSelectedAnalysis}
                                    disabled={!canRunSelected || isLoading}
                                    className="mt-1 w-full px-3 py-2 rounded-md text-xs sm:text-sm font-semibold border shadow-sm disabled:opacity-60"
                                    style={{
                                        background: 'var(--accent)',
                                        borderColor: 'var(--accent)',
                                        color: 'var(--background)',
                                    }}
                                >
                                    {isLoading ? 'Running…' : 'Run'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {error && !processingError && (
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

                <ProcessingErrorModal
                    open={isErrorModalOpen && !!processingError}
                    error={processingError}
                    onClose={() => setIsErrorModalOpen(false)}
                    currentArxivUrl={absUrl}
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
                            stats={stats}
                            onReportGraph={() => {
                                setFeedbackScope('graph');
                                setFeedbackNodeId(null);
                                setFeedbackContextLabel(undefined);
                                setFeedbackOpen(true);
                            }}
                        />
                    </div>
                </div>

                {definitionBank.length > 0 && (
                    <div className="w-full mt-4">
                        <DefinitionBankCard definitions={definitionBank} />
                    </div>
                )}

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
