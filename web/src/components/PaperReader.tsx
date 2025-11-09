"use client";

import React, {
    useEffect,
    useMemo,
    useRef,
    useState,
    forwardRef,
    useImperativeHandle,
} from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import type { ArtifactAnchorIndex } from "@/lib/types";
/* Intentionally avoid importing PDFDocumentProxy directly from pdfjs-dist to prevent
   type mismatches with react-pdf's bundled pdfjs-dist instance. */

// Configure pdf.js worker (ESM)
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    "react-pdf/node_modules/pdfjs-dist/build/pdf.worker.min.mjs",
    import.meta.url
).toString();

type Props = {
    arxivId: string;
    pdfUrl: string;
    // If deep anchors are available (Phase 2), we can use them directly
    anchors?: ArtifactAnchorIndex | null;
    // Phase 1 heuristic search data: map artifactId -> list of search strings
    artifactLookup?: Record<string, string[]>;
    // Controlled selection for automatic scrolling/highlight
    selectedArtifactId?: string | null;
    // Click handler for future reader->graph selection
    onArtifactClick?: (artifactId: string) => void;
    onLoaded?: () => void;
    // Fixed height for the reader container; inner content scrolls within this box
    height?: number | string;
};

export type PaperReaderHandle = {
    scrollToArtifact: (id: string) => Promise<void>;
    pulseArtifact: (id: string) => void;
    clearHighlights: () => void;
};

function normalize(q: string) {
    return q.toLowerCase().replace(/\s+/g, " ").trim();
}

const HIGHLIGHT_MS = 1200;
const ALIGN_OFFSET = 64; // pixels from top when aligning artifacts
const PROOF_BIAS_UP = 240; // additional upward bias when anchor looks like a proof
const CENTER_BIAS_UP = 24; // nudge center slightly upward so heading is more visible
const MIN_SCROLL_DELTA = 8; // px; avoid micro-scroll jitter

const PaperReader = forwardRef<PaperReaderHandle, Props>(function PaperReader(
    { arxivId, pdfUrl, anchors, artifactLookup, selectedArtifactId, onArtifactClick, onLoaded, height = "68vh" },
    ref
) {
    const scrollRef = useRef<HTMLDivElement | null>(null);
    const pdfDocRef = useRef<any>(null);
    const [numPages, setNumPages] = useState<number>(0);
    const [scale, setScale] = useState<number>(1.2);
    // Document URL with fallback to direct arXiv if proxy 404s
    const fallbackUrl = useMemo(() => `https://arxiv.org/pdf/${arxivId}.pdf`, [arxivId]);
    const [docUrl, setDocUrl] = useState<string>(pdfUrl);
    useEffect(() => {
        setDocUrl(pdfUrl);
    }, [pdfUrl, arxivId]);

    // Cache page text for heuristics (page -> normalized text)
    const pageTextCache = useRef<Map<number, string>>(new Map());
    // Refs to page containers for scrolling
    const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());
    // Current highlight state
    const [highlight, setHighlight] = useState<{ page?: number; id?: string } | null>(null);
    // Cached page viewport dimensions (scale=1) for overlay sizing
    const [pageDims, setPageDims] = useState<Record<number, { w: number; h: number }>>({});

    const hasAnchors = !!anchors && Object.keys(anchors || {}).length > 0;

    // Keep a small memory of where we last matched to bias search direction
    const lastMatchPage = useRef<number | null>(null);

    const setPageRef = (pageNumber: number, el: HTMLDivElement | null) => {
        if (!el) {
            pageRefs.current.delete(pageNumber);
        } else {
            pageRefs.current.set(pageNumber, el);
        }
    };

    const onDocLoadSuccess = (doc: any) => {
        pdfDocRef.current = doc;
        setNumPages(doc.numPages);
        // Clear previous caches when changing docs
        pageTextCache.current.clear();
        lastMatchPage.current = null;
        setPageDims({});
        (async () => {
            try {
                const dims: Record<number, { w: number; h: number }> = {};
                for (let p = 1; p <= doc.numPages; p++) {
                    try {
                        const page = await doc.getPage(p);
                        const vp = page.getViewport({ scale: 1 });
                        dims[p] = { w: vp.width, h: vp.height };
                    } catch {
                        // ignore individual page failures
                    }
                }
                setPageDims(dims);
            } catch {
                // ignore
            }
        })();
        onLoaded?.();
    };

    const ensurePageText = async (pageNumber: number): Promise<string> => {
        const cached = pageTextCache.current.get(pageNumber);
        if (cached) return cached;
        const pdf = pdfDocRef.current;
        if (!pdf) return "";
        try {
            const page = await pdf.getPage(pageNumber);
            const content = await page.getTextContent();
            const str = content.items
                .map((it: any) => (typeof it?.str === "string" ? it.str : ""))
                .join(" ");
            const norm = normalize(str);
            pageTextCache.current.set(pageNumber, norm);
            return norm;
        } catch {
            return "";
        }
    };

    const scrollToPage = (pageNumber: number) => {
        const container = scrollRef.current;
        const pageEl = pageRefs.current.get(pageNumber);
        if (!container || !pageEl) return;
        const top = pageEl.offsetTop - 8; // small padding
        container.scrollTo({ top, behavior: "smooth" });
    };

    // Heuristic: from a list of query strings, find the first page containing any query.
    const findPageByQueries = async (queries: string[]): Promise<number | null> => {
        if (!numPages) return null;
        const normQs = Array.from(
            new Set(
                queries
                    .filter(Boolean)
                    .map((q) => normalize(q))
                    .filter((q) => q.length > 1)
            )
        );
        if (normQs.length === 0) return null;

        // Bias search direction near the last match page if available
        const order: number[] = [];
        const last = lastMatchPage.current;
        if (last && last >= 1 && last <= numPages) {
            // Expand outward: last, last+1, last-1, last+2, last-2, ...
            const visited = new Set<number>();
            let delta = 0;
            while (order.length < numPages) {
                const up = last + delta;
                const down = last - delta;
                if (up >= 1 && up <= numPages && !visited.has(up)) {
                    order.push(up);
                    visited.add(up);
                }
                if (down >= 1 && down <= numPages && !visited.has(down)) {
                    order.push(down);
                    visited.add(down);
                }
                delta += 1;
                if (delta > numPages && order.length >= numPages) break;
            }
        } else {
            for (let i = 1; i <= numPages; i++) order.push(i);
        }

        for (const p of order) {
            const text = await ensurePageText(p);
            if (!text) continue;
            for (const q of normQs) {
                if (q && text.includes(q)) {
                    return p;
                }
            }
        }
        return null;
    };

    // Build robust query candidates for all artifact types using anchor metadata and caller-provided hints.
    const buildQueries = (a: { text?: string; type?: string }, hints: string[]): string[] => {
        const qs: string[] = [];
        const push = (s?: string | null) => {
            if (!s) return;
            const t = normalize(String(s));
            if (t.length > 1) qs.push(t);
        };

        // Start with provided hints (display_name, label, etc.)
        (hints || []).forEach((h) => push(h));

        // Add anchor text if present (e.g., "Theorem 3.4", "Definition 2.1", "Eq. (1)")
        if (a?.text) push(a.text);

        // Type synonyms to cover various headings
        const type = (a?.type || "").toLowerCase();
        const typeSyns: Record<string, string[]> = {
            theorem: ["theorem", "thm"],
            definition: ["definition", "def"],
            lemma: ["lemma"],
            proposition: ["proposition", "prop"],
            corollary: ["corollary", "cor"],
            remark: ["remark", "rmk"],
            example: ["example", "ex"],
            conjecture: ["conjecture"],
            claim: ["claim"],
            fact: ["fact"],
            observation: ["observation", "obs"],
            equation: ["equation", "eq"],
            figure: ["figure", "fig"],
            table: ["table", "tab"],
            unknown: [],
        };
        const syns = typeSyns[type] || [];
        syns.forEach((s) => push(s));

        // If anchor.text has a number like "3.4" or "(1)", synthesize useful variants "theorem 3.4", "eq (1)"
        if (a?.text) {
            // Extract number tokens: "3.4" or "12" or "(7)"
            const numMatch = a.text.match(/(\(?\d+(?:\.\d+)*\)?)/);
            if (numMatch) {
                const num = numMatch[1]; // keep parentheses if present
                syns.forEach((s) => push(`${s} ${num}`));
                // Also try "eq (n)" or "equation (n)"
                if (type === "equation") {
                    push(`eq ${num}`);
                    push(`equation ${num}`);
                }
                if (type === "figure") {
                    push(`fig ${num}`);
                    push(`figure ${num}`);
                }
                if (type === "table") {
                    push(`tab ${num}`);
                    push(`table ${num}`);
                }
            }
        }

        // De-duplicate while preserving order
        const seen = new Set<string>();
        const out: string[] = [];
        for (const q of qs) {
            if (!seen.has(q)) {
                seen.add(q);
                out.push(q);
            }
        }
        return out.slice(0, 8); // keep it small
    };

    // Try to find a better anchor using the rendered text layer spans.
    // Returns the top coordinate in the scroll container if found, else null.
    const findTextTop = (pageNumber: number, queries: string[]): number | null => {
        const container = scrollRef.current;
        const pageEl = pageRefs.current.get(pageNumber) || null;
        if (!container || !pageEl) return null;

        // The text layer is rendered by react-pdf as .react-pdf__Page__textContent with many spans.
        const textLayer = pageEl.querySelector(".react-pdf__Page__textContent");
        if (!textLayer) return null;

        const normQs = Array.from(
            new Set(
                (queries || [])
                    .filter(Boolean)
                    .map((q) => normalize(q))
                    .filter((q) => q.length > 1)
            )
        );
        if (!normQs.length) return null;

        const spans = Array.from(textLayer.querySelectorAll("span"));
        let bestTop: number | null = null;

        for (const span of spans) {
            const txt = normalize(span.textContent || "");
            // Prefer the first query that matches; label/display_name should be first in queries.
            const hit = normQs.find((q) => q && txt.includes(q));
            if (!hit) continue;

            const spanRect = (span as HTMLElement).getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            // Translate browser viewport coords to scroll container coords
            const topInContainer = container.scrollTop + (spanRect.top - containerRect.top);

            bestTop = topInContainer;
            break;
        }

        return bestTop;
    };

    // Convert anchor bbox to pixel units at current scale and page dims.
    // If bbox appears normalized (0..1), compute px using pageDims; otherwise treat as px at scale=1 and multiply by scale.
    // Optionally choose between top-origin and bottom-origin normalized y by comparing to a refinedTopInContainer if provided.
    const bboxPx = (
        pageNumber: number,
        bbox: [number, number, number, number],
        opts?: { refinedTopInContainer?: number; pageOffsetTop?: number }
    ): { left: number; top: number; width: number; height: number } => {
        const dims = pageDims[pageNumber];
        if (!dims) return { left: 0, top: 0, width: 0, height: 0 };
        const [bx, by, bw, bh] = bbox;
        const pageW = dims.w * scale;
        const pageH = dims.h * scale;
        const normalized =
            bx >= 0 && by >= 0 && bw >= 0 && bh >= 0 &&
            bx <= 1.05 && by <= 1.05 && bw <= 1.05 && bh <= 1.05;

        if (normalized) {
            const left = bx * pageW;
            const width = bw * pageW;
            const topTop = by * pageH; // assume y from top
            const topBottom = pageH - (by + bh) * pageH; // assume y from bottom

            if (opts?.refinedTopInContainer != null && opts?.pageOffsetTop != null) {
                const candA = opts.pageOffsetTop + topTop;
                const candB = opts.pageOffsetTop + topBottom;
                const diffA = Math.abs(candA - opts.refinedTopInContainer);
                const diffB = Math.abs(candB - opts.refinedTopInContainer);
                const top = diffA <= diffB ? topTop : topBottom;
                const height = bh * pageH;
                return { left, top, width, height };
            } else {
                // default to top-origin interpretation
                const top = topTop;
                const height = bh * pageH;
                return { left, top, width, height };
            }
        } else {
            // pixels at scale=1 → multiply by scale
            return {
                left: bx * scale,
                top: by * scale,
                width: bw * scale,
                height: bh * scale,
            };
        }
    };

    const scrollToArtifact = async (id: string) => {
        // Phase 2: use anchors for precise page and bbox
        if (hasAnchors && anchors) {
            const a = anchors[id];
            if (a) {
                lastMatchPage.current = a.page;
                const container = scrollRef.current;
                const pageEl = pageRefs.current.get(a.page) || null;
                if (container && pageEl) {
                    const hints = (artifactLookup?.[id] || []);
                    const q = buildQueries(a as any, hints);
                    const refinedTop = findTextTop(a.page, q);

                    const px = bboxPx(a.page, a.bbox, {
                        refinedTopInContainer: refinedTop ?? undefined,
                        pageOffsetTop: pageEl.offsetTop,
                    });
                    const rectTop = pageEl.offsetTop + px.top;
                    const rectHeight = px.height;
                    const rectBottom = rectTop + rectHeight;
                    const viewTop = container.scrollTop;
                    const viewHeight = container.clientHeight;
                    const viewBottom = viewTop + viewHeight;

                    let targetTop: number | null = null;

                    if (rectTop >= viewTop + 8 && rectBottom <= viewBottom - 8) {
                        // fully visible, do nothing
                        targetTop = null;
                    } else {
                        // If we found a refined text hit, align to it; otherwise use center/top heuristics.
                        if (typeof refinedTop === "number") {
                            targetTop = refinedTop - ALIGN_OFFSET;
                        } else if (rectHeight >= viewHeight * 0.85) {
                            targetTop = rectTop - ALIGN_OFFSET; // tall block → show its top
                        } else {
                            targetTop = rectTop + rectHeight / 2 - viewHeight / 2 - CENTER_BIAS_UP; // center with slight up-bias
                        }
                    }

                    // If anchor looks like a proof, bias further upward to land near the statement header.
                    const looksLikeProof =
                        (typeof a.text === "string" && /\bproof\b/i.test(a.text)) ||
                        (a as any)?.type === "proof";
                    if (targetTop !== null) {
                        if (looksLikeProof) targetTop -= PROOF_BIAS_UP;
                        // Clamp within the page bounds so we don't overscroll and avoid micro-jitter
                        const pageH = (pageDims[a.page]?.h || 0) * scale;
                        const maxScroll = Math.max(0, pageEl.offsetTop + pageH - viewHeight);
                        const clampedTop = Math.max(0, Math.min(targetTop, maxScroll));
                        if (Math.abs(clampedTop - viewTop) >= MIN_SCROLL_DELTA) {
                            container.scrollTo({ top: clampedTop, behavior: "smooth" });
                        }
                    } else {
                        // already visible; no scroll
                    }
                } else {
                    // Fallback to page-level scroll if page ref not yet available
                    scrollToPage(a.page);
                }
                setHighlight({ page: a.page, id });
                window.setTimeout(() => setHighlight(null), HIGHLIGHT_MS);
                return;
            }
        }

        // Phase 1: heuristic search using artifactLookup text
        const queries = artifactLookup?.[id] || [];
        const page = await findPageByQueries(queries);
        if (page) {
            lastMatchPage.current = page;
            const container = scrollRef.current;
            const pageEl = pageRefs.current.get(page) || null;
            if (container && pageEl) {
                // Calm alignment near the top of the page to avoid jumping to page top
                const targetTop = Math.max(0, pageEl.offsetTop - ALIGN_OFFSET);
                container.scrollTo({ top: targetTop, behavior: "smooth" });
            } else {
                scrollToPage(page);
            }
            setHighlight({ page, id });
            window.setTimeout(() => setHighlight(null), HIGHLIGHT_MS);
            return;
        }

        // Fallback: scroll to top if nothing found
        const container = scrollRef.current;
        if (container) container.scrollTo({ top: 0, behavior: "smooth" });
    };

    const pulseArtifact = (id: string) => {
        // If anchor exists, pulse that page; else pulse lastMatch or page 1
        if (hasAnchors && anchors?.[id]) {
            const p = anchors[id].page;
            setHighlight({ page: p, id });
            window.setTimeout(() => setHighlight(null), HIGHLIGHT_MS);
        } else {
            const p = lastMatchPage.current || 1;
            setHighlight({ page: p });
            window.setTimeout(() => setHighlight(null), HIGHLIGHT_MS);
        }
    };

    const clearHighlights = () => setHighlight(null);

    useImperativeHandle(
        ref,
        () => ({
            scrollToArtifact,
            pulseArtifact,
            clearHighlights,
        }),
        [anchors, artifactLookup, numPages]
    );

    // Auto-scroll when selection changes
    useEffect(() => {
        if (selectedArtifactId) {
            scrollToArtifact(selectedArtifactId);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedArtifactId, arxivId]);

    const zoomIn = () => setScale((s) => Math.min(2.0, s + 0.1));
    const zoomOut = () => setScale((s) => Math.max(0.6, s - 0.1));
    const fitWidth = () => setScale(1.2);

    return (
        <div
            className="flex flex-col text-[var(--text)]"
            style={{ height: typeof height === "number" ? `${height}px` : height }}
        >

            <div
                ref={scrollRef}
                className="flex-1 overflow-auto py-2"
                style={{ scrollBehavior: "smooth" }}
            >
                <Document
                    file={docUrl}
                    onLoadSuccess={onDocLoadSuccess}
                    loading={<div className="text-sm text-gray-600 px-2">Loading PDF…</div>}
                    error={
                        <div className="text-sm text-red-600 px-2">
                            Failed to load PDF.
                            {docUrl !== fallbackUrl ? (
                                <span className="ml-2">Retrying with arXiv URL…</span>
                            ) : null}
                        </div>
                    }
                    onLoadError={() => {
                        // Fallback to direct arXiv PDF if proxy is unavailable
                        if (docUrl !== fallbackUrl) {
                            setDocUrl(fallbackUrl);
                        }
                    }}
                >
                    {Array.from({ length: numPages }, (_, idx) => idx + 1).map((pageNumber) => {
                        const isHighlighted = highlight?.page === pageNumber;
                        return (
                            <div
                                key={pageNumber}
                                ref={(el) => setPageRef(pageNumber, el)}
                                className="relative mx-auto my-2 rounded-md"
                                style={{
                                    width: "fit-content",
                                    outline: isHighlighted ? "3px solid rgba(59,130,246,0.7)" : "none",
                                    transition: "outline-color 200ms",
                                }}
                            >
                                {/* Overlay pulse indicator */}
                                {isHighlighted ? (
                                    <div
                                        aria-hidden
                                        style={{
                                            position: "absolute",
                                            inset: 0,
                                            borderRadius: "0.375rem",
                                            boxShadow: "0 0 0 4px rgba(59,130,246,0.35) inset",
                                            pointerEvents: "none",
                                            animation: "pulse 1.8s ease-in-out 1",
                                        }}
                                    />
                                ) : null}
                                <Page
                                    pageNumber={pageNumber}
                                    scale={scale}
                                    renderAnnotationLayer={false}
                                    renderTextLayer={true}
                                />
                                {anchors && pageDims[pageNumber] ? (
                                    <div
                                        aria-hidden
                                        style={{
                                            position: "absolute",
                                            left: 0,
                                            top: 0,
                                            width: pageDims[pageNumber].w * scale,
                                            height: pageDims[pageNumber].h * scale,
                                            pointerEvents: "none",
                                            zIndex: 10,
                                        }}
                                    >
                                        {Object.values(anchors)
                                            .filter((a) => a.page === pageNumber)
                                            .map((a) => {
                                                const isSel = highlight?.id === a.id;
                                                const px = bboxPx(pageNumber, a.bbox);
                                                return (
                                                    <div
                                                        key={a.id}
                                                        title={a.text || a.id}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            // Visibility-first scroll with calm top/center alignment using px bbox.
                                                            const container = scrollRef.current;
                                                            const pageEl = pageRefs.current.get(pageNumber) || null;
                                                            if (container && pageEl) {
                                                                const rectTop = pageEl.offsetTop + px.top;
                                                                const rectHeight = px.height;
                                                                const rectBottom = rectTop + rectHeight;
                                                                const viewTop = container.scrollTop;
                                                                const viewHeight = container.clientHeight;
                                                                const viewBottom = viewTop + viewHeight;
                                                                let targetTop: number | null = null;

                                                                if (rectTop >= viewTop + 8 && rectBottom <= viewBottom - 8) {
                                                                    targetTop = null; // already visible
                                                                } else {
                                                                    // Try refinement for overlays too
                                                                    const q = buildQueries(a as any, []);
                                                                    const hitTop = findTextTop(pageNumber, q);
                                                                    if (typeof hitTop === "number") {
                                                                        targetTop = hitTop - ALIGN_OFFSET;
                                                                    } else if (rectHeight >= viewHeight * 0.85) {
                                                                        targetTop = rectTop - ALIGN_OFFSET;
                                                                    } else {
                                                                        targetTop = rectTop + rectHeight / 2 - viewHeight / 2 - CENTER_BIAS_UP;
                                                                    }
                                                                }

                                                                if (targetTop !== null) {
                                                                    const pageH = (pageDims[pageNumber]?.h || 0) * scale;
                                                                    const maxScroll = Math.max(0, pageEl.offsetTop + pageH - viewHeight);
                                                                    const clampedTop = Math.max(0, Math.min(targetTop, maxScroll));
                                                                    if (Math.abs(clampedTop - viewTop) >= MIN_SCROLL_DELTA) {
                                                                        container.scrollTo({ top: clampedTop, behavior: "smooth" });
                                                                    }
                                                                }
                                                            }
                                                            onArtifactClick?.(a.id);
                                                            setHighlight({ page: pageNumber, id: a.id });
                                                        }}
                                                        style={{
                                                            position: "absolute",
                                                            left: px.left,
                                                            top: px.top,
                                                            width: px.width,
                                                            height: px.height,
                                                            border: isSel
                                                                ? "2px solid rgba(59,130,246,0.9)"
                                                                : "1.5px solid rgba(59,130,246,0.6)",
                                                            background: isSel
                                                                ? "rgba(59,130,246,0.15)"
                                                                : "rgba(59,130,246,0.08)",
                                                            borderRadius: 4,
                                                            cursor: "pointer",
                                                            pointerEvents: "auto",
                                                        }}
                                                    />
                                                );
                                            })}
                                    </div>
                                ) : null}
                            </div>
                        );
                    })}
                </Document>
            </div>

            {/* Simple keyframes for pulse */}
            <style jsx global>{`
        @keyframes pulse {
          0% {
            opacity: 0.0;
          }
          20% {
            opacity: 1.0;
          }
          100% {
            opacity: 0.0;
          }
        }
        /* Keep text layer invisible and non-interactive, but usable for measurement */
        .react-pdf__Page__textContent {
          opacity: 0;
          pointer-events: none;
        }
      `}</style>
        </div>
    );
});

export default PaperReader;
