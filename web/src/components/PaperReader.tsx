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
    // Full statement text used for regex-based matching (normalized upstream)
    selectedArtifactText?: string | null;
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
const HEADING_SEARCH_WINDOW = 320; // px above anchor to search for "Definition/Lemma/Theorem" heading
const COLUMN_TOLERANCE = 96; // px tolerance to stay in the same column as the anchor
const MAX_PARA_SEARCH_WINDOW = 900; // px below heading to search for the end of the statement paragraph
const LINE_JOIN_TOL = 2; // px tolerance to merge spans into one line
const LINE_BREAK_GAP = 28; // px gap between lines that indicates a paragraph break
const HEADING_WORDS = [
    "proof",
    "theorem",
    "lemma",
    "definition",
    "proposition",
    "corollary",
    "remark",
    "example",
    "claim",
    "fact",
    "observation",
    "conjecture",
];

const PaperReader = forwardRef<PaperReaderHandle, Props>(function PaperReader(
    { arxivId, pdfUrl, anchors, artifactLookup, selectedArtifactId, selectedArtifactText, onArtifactClick, onLoaded, height = "68vh" },
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

    // Text-layer span extraction and highlight range computation
    type SpanInfo = { el: HTMLSpanElement; textNorm: string; left: number; top: number; width: number; height: number };
    const getTextSpans = (pageNumber: number): SpanInfo[] => {
        const pageEl = pageRefs.current.get(pageNumber) || null;
        if (!pageEl) return [];
        const textLayer = pageEl.querySelector(".react-pdf__Page__textContent");
        if (!textLayer) return [];
        const pageRect = pageEl.getBoundingClientRect();
        const spans = Array.from(textLayer.querySelectorAll("span"));
        return spans.map((s) => {
            const el = s as HTMLSpanElement;
            const rect = el.getBoundingClientRect();
            return {
                el,
                textNorm: normalize(el.textContent || ""),
                left: rect.left - pageRect.left,
                top: rect.top - pageRect.top,
                width: rect.width,
                height: rect.height,
            };
        });
    };
    const findHeadingTopInSameColumn = (
        pageNumber: number,
        px: { left: number; top: number; width: number; height: number },
        queries: string[]
    ): number | null => {
        const spans = getTextSpans(pageNumber);
        if (!spans.length) return null;
        const normQs = Array.from(new Set((queries || []).map((q) => normalize(q)).filter((q) => q.length > 1)));
        if (!normQs.length) return null;
        const colTol = Math.max(COLUMN_TOLERANCE, px.width * 0.75);
        const windowTop = Math.max(0, px.top - HEADING_SEARCH_WINDOW);
        const candidates = spans
            .filter(
                (sp) =>
                    sp.top < px.top &&
                    sp.top >= windowTop &&
                    Math.abs(sp.left - px.left) <= colTol &&
                    normQs.some((q) => sp.textNorm.includes(q))
            )
            .sort((a, b) => b.top - a.top);
        if (candidates.length) return candidates[0].top;
        return null;
    };
    const computeHighlightRange = (
        pageNumber: number,
        px: { left: number; top: number; width: number; height: number },
        queries: string[]
    ): { top: number; bottom: number; left: number; right: number } => {
        // 1) Resolve start at heading if available, else use anchor top
        const start = findHeadingTopInSameColumn(pageNumber, px, queries);
        const top = start != null ? start : px.top;

        // 2) Try to extend downwards to the end of the statement paragraph using text spans in same column
        const spans = getTextSpans(pageNumber)
            .filter((sp) => {
                const withinCol = Math.abs(sp.left - px.left) <= Math.max(COLUMN_TOLERANCE, px.width * 0.75);
                const belowStart = sp.top + sp.height >= top - LINE_JOIN_TOL;
                const withinWindow = sp.top <= top + MAX_PARA_SEARCH_WINDOW;
                return withinCol && belowStart && withinWindow;
            })
            .sort((a, b) => a.top - b.top);

        // Group spans into line blocks by vertical proximity
        type Line = { top: number; bottom: number; text: string };
        const lines: Line[] = [];
        for (const sp of spans) {
            const spBottom = sp.top + sp.height;
            const last = lines[lines.length - 1];
            if (!last || Math.abs(sp.top - last.top) > LINE_JOIN_TOL) {
                // new line
                lines.push({ top: sp.top, bottom: spBottom, text: sp.textNorm });
            } else {
                // merge into current line
                last.bottom = Math.max(last.bottom, spBottom);
                if (sp.textNorm) {
                    last.text = last.text ? `${last.text} ${sp.textNorm}` : sp.textNorm;
                }
            }
        }

        // Walk lines until we hit a paragraph break or a new heading (e.g., "Proof", "Theorem", ...)
        let bottom = Math.max(px.top + px.height, top + 8);
        let prevLine: Line | null = null;
        const isHeading = (t: string) => {
            const txt = t.toLowerCase().trim();
            return /^(proof|theorem|lemma|definition|proposition|corollary|remark|example|claim|fact|observation|conjecture)\b/.test(txt);
        };

        for (const line of lines) {
            // Skip any lines that are above the starting top (robust to tiny overlaps)
            if (line.bottom < top - LINE_JOIN_TOL) {
                prevLine = line;
                continue;
            }

            // If we encounter a new heading line after we started the statement, stop before it
            if (prevLine && isHeading(line.text)) {
                break;
            }

            // If there is a large vertical gap, treat it as paragraph end and stop before current line
            if (prevLine && line.top - prevLine.bottom > LINE_BREAK_GAP) {
                break;
            }

            bottom = Math.max(bottom, line.bottom);
            prevLine = line;
        }

        // Clamp bottom within the page
        const pageH = (pageDims[pageNumber]?.h || 0) * scale;
        if (pageH > 0) {
            bottom = Math.min(bottom, pageH - 2);
        }

        // Width: expand to cover the text column based on spans between [top, bottom]
        let left = px.left;
        let right = px.left + px.width;
        if (spans.length) {
            const used = spans.filter((sp) => sp.top <= bottom && sp.top + sp.height >= top - LINE_JOIN_TOL);
            if (used.length) {
                left = Math.min(left, ...used.map((sp) => sp.left));
                right = Math.max(right, ...used.map((sp) => sp.left + sp.width));
            }
        }

        return { top, bottom, left, right };
    };

    // Phase 1 (no anchors): compute highlight purely from spans on the page
    const computeHighlightRangeFromSpans = (
        pageNumber: number,
        rawQueries: string[]
    ): { top: number; bottom: number; left: number; right: number } | null => {
        const spans = getTextSpans(pageNumber);
        if (!spans.length) return null;

        const normQs = Array.from(new Set((rawQueries || []).map((q) => normalize(q)).filter((q) => q.length > 1)));
        const headingRe = /^(proof|theorem|lemma|definition|proposition|corollary|remark|example|claim|fact|observation|conjecture)\b/;

        // Find a span matching any of the queries to infer the column
        let querySpan: SpanInfo | null = null;
        if (normQs.length) {
            querySpan = spans.find((sp) => normQs.some((q) => sp.textNorm.includes(q))) || null;
        }

        // Try to find the nearest heading above the query span in the same column
        let headingSpan: SpanInfo | null = null;
        if (querySpan) {
            const windowTop = Math.max(0, querySpan.top - HEADING_SEARCH_WINDOW);
            const colLeft = querySpan.left;
            const colTol = Math.max(COLUMN_TOLERANCE, 80);
            const candidates = spans
                .filter(
                    (sp) =>
                        sp.top < querySpan!.top &&
                        sp.top >= windowTop &&
                        Math.abs(sp.left - colLeft) <= colTol &&
                        headingRe.test(sp.textNorm.trim())
                )
                .sort((a, b) => b.top - a.top);
            headingSpan = candidates[0] || null;
        }
        // Fallback: any heading on the page
        if (!headingSpan) {
            const heads = spans.filter((sp) => headingRe.test(sp.textNorm.trim())).sort((a, b) => a.top - b.top);
            headingSpan = heads[0] || null;
        }

        // Determine start top and column
        const top = headingSpan ? headingSpan.top : querySpan ? querySpan.top : spans[0].top;
        const colLeft = querySpan ? querySpan.left : headingSpan ? headingSpan.left : spans[0].left;
        const colTol = Math.max(COLUMN_TOLERANCE, 80);

        // Collect spans in the same column within window
        const usable = spans
            .filter(
                (sp) =>
                    Math.abs(sp.left - colLeft) <= colTol &&
                    sp.top + sp.height >= top - LINE_JOIN_TOL &&
                    sp.top <= top + MAX_PARA_SEARCH_WINDOW
            )
            .sort((a, b) => a.top - b.top);

        // Group into lines
        type Line = { top: number; bottom: number; text: string };
        const lines: Line[] = [];
        for (const sp of usable) {
            const spBottom = sp.top + sp.height;
            const last = lines[lines.length - 1];
            if (!last || Math.abs(sp.top - last.top) > LINE_JOIN_TOL) {
                lines.push({ top: sp.top, bottom: spBottom, text: sp.textNorm });
            } else {
                last.bottom = Math.max(last.bottom, spBottom);
                if (sp.textNorm) {
                    last.text = last.text ? `${last.text} ${sp.textNorm}` : sp.textNorm;
                }
            }
        }

        // Walk until paragraph end or next heading
        let bottom = top + 8;
        let prevLine: Line | null = null;
        for (const line of lines) {
            if (line.bottom < top - LINE_JOIN_TOL) {
                prevLine = line;
                continue;
            }
            if (prevLine && headingRe.test(line.text.trim())) {
                break;
            }
            if (prevLine && line.top - prevLine.bottom > LINE_BREAK_GAP) {
                break;
            }
            bottom = Math.max(bottom, line.bottom);
            prevLine = line;
        }

        // Clamp bottom within page
        const pageH = (pageDims[pageNumber]?.h || 0) * scale;
        if (pageH > 0) {
            bottom = Math.min(bottom, pageH - 2);
        }

        // Expand width to cover column
        let left = colLeft;
        let right = colLeft + 8;
        if (usable.length) {
            const used = usable.filter((sp) => sp.top <= bottom && sp.top + sp.height >= top - LINE_JOIN_TOL);
            if (used.length) {
                left = Math.min(...used.map((sp) => sp.left));
                right = Math.max(...used.map((sp) => sp.left + sp.width));
            }
        }

        return { top, bottom, left, right };
    };

    // ===== Regex-based statement matching helpers =====
    const normalizeForIndex = (s: string): string => {
        if (!s) return "";
        // Normalize ligatures and punctuation, then collapse to words
        return s
            .replace(/\uFB01/g, "fi")
            .replace(/\uFB02/g, "fl")
            .replace(/[\u2018\u2019]/g, "'")
            .replace(/[\u201C\u201D]/g, '"')
            .replace(/[\u2013\u2014]/g, "-")
            .toLowerCase()
            .replace(/[^a-z0-9]+/gi, " ")
            .trim()
            .replace(/\s+/g, " ");
    };
    const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const buildStatementRegex = (text: string): RegExp | null => {
        const norm = normalizeForIndex(text || "");
        if (!norm) return null;
        const tokens = norm.split(/\s+/).filter((w) => w.length > 1).slice(0, 120);
        if (!tokens.length) return null;
        const pattern = tokens.map(escapeRegExp).join("\\s+");
        try {
            return new RegExp(pattern, "im");
        } catch {
            return null;
        }
    };
    const buildPageIndex = (
        pageNumber: number
    ): { text: string; segments: Array<{ start: number; end: number; left: number; right: number; top: number; bottom: number }> } | null => {
        const spans = getTextSpans(pageNumber);
        if (!spans.length) return null;
        const segments: Array<{ start: number; end: number; left: number; right: number; top: number; bottom: number }> = [];
        let text = "";
        let pos = 0;
        for (const sp of spans) {
            const raw = (sp.el.textContent || "").toString();
            const token = normalizeForIndex(raw);
            if (!token) continue;
            if (text.length > 0) {
                text += " ";
                pos += 1;
            }
            const start = pos;
            text += token;
            pos += token.length;
            segments.push({
                start,
                end: pos,
                left: sp.left,
                right: sp.left + sp.width,
                top: sp.top,
                bottom: sp.top + sp.height,
            });
        }
        return { text, segments };
    };
    const findRegexMatchBounds = (
        pageNumber: number,
        statementText?: string | null
    ): { top: number; bottom: number; left: number; right: number } | null => {
        if (!statementText) return null;
        const re = buildStatementRegex(statementText);
        if (!re) return null;
        const idx = buildPageIndex(pageNumber);
        if (!idx) return null;
        const m = re.exec(idx.text);
        if (!m || m.index == null) return null;
        const start = m.index;
        const end = start + (m[0]?.length || 0);
        if (end <= start) return null;
        let firstIdx = -1;
        let lastIdx = -1;
        for (let i = 0; i < idx.segments.length; i++) {
            const sg = idx.segments[i];
            if (firstIdx === -1 && sg.end > start) firstIdx = i;
            if (sg.start < end) lastIdx = i;
        }
        if (firstIdx === -1 || lastIdx === -1) return null;
        const involved = idx.segments.slice(firstIdx, lastIdx + 1);
        const left = Math.min(...involved.map((s) => s.left));
        const right = Math.max(...involved.map((s) => s.right));
        const top = involved[0].top;
        const bottom = involved[involved.length - 1].bottom;
        return { top, bottom, left, right };
    };

    // Two-anchor regex: match first K tokens for start and last K tokens for end; compute union band
    const findRegexBandTwoAnchors = (
        pageNumber: number,
        statementText?: string | null,
        opts?: { kStart?: number; kEnd?: number }
    ): { top: number; bottom: number; left: number; right: number } | null => {
        if (!statementText) return null;
        const idx = buildPageIndex(pageNumber);
        if (!idx) return null;

        const norm = normalizeForIndex(statementText);
        if (!norm) return null;
        const tokens = norm.split(/\s+/).filter(Boolean);
        if (tokens.length < 4) return null;

        const kS = Math.min(opts?.kStart ?? 15, Math.max(2, Math.ceil(tokens.length * 0.2)));
        const kE = Math.min(opts?.kEnd ?? 15, Math.max(2, Math.ceil(tokens.length * 0.2)));

        const startTokens = tokens.slice(0, kS);
        const endTokens = tokens.slice(-kE);

        const reStart = new RegExp(startTokens.map(escapeRegExp).join("\\s+"), "im");
        const reEnd = new RegExp(endTokens.map(escapeRegExp).join("\\s+"), "im");

        const mS = reStart.exec(idx.text);
        const mE = reEnd.exec(idx.text);

        if (!mS && !mE) return null;

        const toSegRange = (startIdx: number, endIdx: number): { first: number; last: number } | null => {
            let first = -1;
            let last = -1;
            for (let i = 0; i < idx.segments.length; i++) {
                const sg = idx.segments[i];
                if (first === -1 && sg.end > startIdx) first = i;
                if (sg.start < endIdx) last = i;
            }
            if (first === -1 || last === -1) return null;
            return { first, last };
        };

        const rangeS = mS ? toSegRange(mS.index, mS.index + (mS[0]?.length || 0)) : null;
        const rangeE = mE ? toSegRange(mE.index, mE.index + (mE[0]?.length || 0)) : null;

        // If both exist and start is before end, union them; else fall back to single range
        let firstIdx = -1;
        let lastIdx = -1;
        if (rangeS && rangeE && mS && mE && mS.index <= mE.index) {
            firstIdx = rangeS.first;
            lastIdx = rangeE.last;
        } else if (rangeS) {
            firstIdx = rangeS.first;
            lastIdx = rangeS.last;
        } else if (rangeE) {
            firstIdx = rangeE.first;
            lastIdx = rangeE.last;
        } else {
            return null;
        }

        const involved = idx.segments.slice(firstIdx, lastIdx + 1);
        if (!involved.length) return null;
        const left = Math.min(...involved.map((s) => s.left));
        const right = Math.max(...involved.map((s) => s.right));
        const top = involved[0].top;
        const bottom = involved[involved.length - 1].bottom;
        return { top, bottom, left, right };
    };
    // ==================================================
    // Paragraph-based simple matching and expansion to paragraph boundaries

    function median(nums: number[]): number {
        if (!nums.length) return 0;
        const arr = [...nums].sort((a, b) => a - b);
        const mid = Math.floor(arr.length / 2);
        return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
    }

    // Build a single lowercased linear text of the page with explicit line and paragraph breaks.
    // - Between spans in same line: single space
    // - Between lines: "\n" for small gaps, "\n\n" for large gaps (paragraph break)
    // Returns the linear text and a mapping of (start,end,rect) for each span.
    const buildPageLinearText = (
        pageNumber: number
    ): {
        textLower: string;
        segs: Array<{ start: number; end: number; left: number; right: number; top: number; bottom: number }>;
    } | null => {
        const spans = getTextSpans(pageNumber);
        if (!spans.length) return null;

        // Sort spans by top then left
        const sorted = [...spans].sort((a, b) => (a.top === b.top ? a.left - b.left : a.top - b.top));

        // Dynamic tolerances
        const heights = sorted.map((s) => s.height || (s as any).bottom ? ((s as any).bottom - s.top) : 0);
        const medH = Math.max(1, median(heights) || 12);
        const lineJoinTol = Math.max(2, 0.35 * medH);

        // Group into lines
        type Line = { spans: ReturnType<typeof getTextSpans>; top: number; bottom: number };
        const lines: Line[] = [];
        let cur: Line | null = null;
        for (const s of sorted) {
            if (!cur) {
                cur = { spans: [s], top: s.top, bottom: s.top + s.height };
                lines.push(cur);
            } else {
                if (Math.abs(s.top - cur.top) <= lineJoinTol) {
                    cur.spans.push(s);
                    cur.bottom = Math.max(cur.bottom, s.top + s.height);
                } else {
                    cur = { spans: [s], top: s.top, bottom: s.top + s.height };
                    lines.push(cur);
                }
            }
        }
        // Sort spans within line by left
        lines.forEach((ln) => (ln.spans = [...ln.spans].sort((a, b) => a.left - b.left)));

        const lineHeights = lines.map((ln) => ln.bottom - ln.top);
        const medLH = Math.max(1, median(lineHeights) || medH);
        const paraBreakGap = Math.max(LINE_BREAK_GAP, 1.4 * medLH);

        let textLower = "";
        const segs: Array<{ start: number; end: number; left: number; right: number; top: number; bottom: number }> = [];
        let pos = 0;

        for (let i = 0; i < lines.length; i++) {
            const ln = lines[i];
            // Append line content
            for (let j = 0; j < ln.spans.length; j++) {
                const sp = ln.spans[j];
                const raw = (sp.el.textContent || "").toString().toLowerCase();
                // add single space between spans in same line
                if (j > 0) {
                    textLower += " ";
                    pos += 1;
                }
                const start = pos;
                textLower += raw;
                pos += raw.length;
                segs.push({
                    start,
                    end: pos,
                    left: sp.left,
                    right: sp.left + sp.width,
                    top: sp.top,
                    bottom: sp.top + sp.height,
                });
            }
            // Append line or paragraph break
            const next = lines[i + 1];
            if (next) {
                const gap = next.top - ln.bottom;
                const br = gap > paraBreakGap ? "\n\n" : "\n";
                textLower += br;
                pos += br.length;
            }
        }

        return { textLower, segs };
    };

    // Build a forgiving regex from the statement preview:
    // - Escape regex metachars
    // - Convert whitespace runs to \s+
    // - Make common punctuation optional (\W*)
    const buildStatementRegexLoose = (text: string): RegExp | null => {
        if (!text) return null;
        // Lowercase src to match our lowercased linear text
        let pat = text.toLowerCase();

        // Escape regex special chars
        pat = pat.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

        // Replace punctuation with optional non-word sequences
        pat = pat.replace(/[.,;:!?\-\u2013\u2014\u2018\u2019\u201c\u201d()\[\]\{\}]/g, "\\W*");

        // Collapse whitespace runs to \s+
        pat = pat.replace(/\s+/g, "\\s+");

        try {
            return new RegExp(pat, "im");
        } catch {
            return null;
        }
    };

    // Find statement match and expand to paragraph boundaries in the page linear text
    const findParagraphBandByPreview = (
        pageNumber: number,
        statementText?: string | null
    ): { top: number; bottom: number; left: number; right: number } | null => {
        if (!statementText) return null;
        const idx = buildPageLinearText(pageNumber);
        if (!idx) return null;

        const re = buildStatementRegexLoose(statementText);
        if (!re) return null;

        const m = re.exec(idx.textLower);
        if (!m || m.index == null) return null;

        // Expand to paragraph boundaries using '\n\n'
        const matchStart = m.index;
        const matchEnd = matchStart + (m[0]?.length || 0);

        let paraStart = idx.textLower.lastIndexOf("\n\n", matchStart);
        paraStart = paraStart === -1 ? 0 : paraStart + 2;

        let paraEnd = idx.textLower.indexOf("\n\n", matchEnd);
        paraEnd = paraEnd === -1 ? idx.textLower.length : paraEnd;

        // Map to overlapping spans
        const involved = idx.segs.filter((sg) => sg.end > paraStart && sg.start < paraEnd);
        if (!involved.length) return null;

        const left = Math.min(...involved.map((s) => s.left));
        const right = Math.max(...involved.map((s) => s.right));
        const top = Math.min(...involved.map((s) => s.top));
        const bottom = Math.max(...involved.map((s) => s.bottom));
        return { top, bottom, left, right };
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
        } else {
            const p = lastMatchPage.current || 1;
            setHighlight({ page: p });
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
                                onClick={() => setHighlight(null)}
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
                                {pageDims[pageNumber] ? (
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
                                        {anchors
                                            ? Object.values(anchors)
                                                .filter((a) => a.page === pageNumber)
                                                .map((a) => {
                                                    const isSel = highlight?.id === a.id;
                                                    const px = bboxPx(pageNumber, a.bbox);
                                                    const q = buildQueries(a as any, artifactLookup?.[a.id] || []);
                                                    // Gate highlighting on transient highlight state only (auto-clears)
                                                    const isActive = !!isSel;
                                                    // Prefer two-anchor regex band; fallback to single regex; then to heuristic
                                                    const rx2 = isActive && selectedArtifactText ? findRegexBandTwoAnchors(pageNumber, selectedArtifactText) : null;
                                                    const rx = !rx2 && isActive && selectedArtifactText ? findRegexMatchBounds(pageNumber, selectedArtifactText) : null;
                                                    const pb = isActive && selectedArtifactText ? findParagraphBandByPreview(pageNumber, selectedArtifactText) : null;
                                                    const band = pb ?? rx2 ?? rx ?? (isActive ? computeHighlightRange(pageNumber, px, q) : null);

                                                    return (
                                                        <React.Fragment key={a.id}>
                                                            {/* Soft background highlight for the selected artifact's statement */}
                                                            {band ? (
                                                                <div
                                                                    aria-hidden
                                                                    style={{
                                                                        position: "absolute",
                                                                        left: Math.max(0, band.left - 2),
                                                                        top: band.top,
                                                                        width: Math.max(8, band.right - band.left) + 4,
                                                                        height: Math.max(8, band.bottom - band.top),
                                                                        background: "rgba(250, 204, 21, 0.24)", // slightly stronger yellow
                                                                        borderRadius: 4,
                                                                        pointerEvents: "none",
                                                                        zIndex: 1,
                                                                    }}
                                                                />
                                                            ) : null}

                                                            {/* Clickable anchor bbox overlay */}
                                                            <div
                                                                key={a.id + "__box"}
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
                                                                            const q2 = buildQueries(a as any, []);
                                                                            const hitTop = findTextTop(pageNumber, q2);
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
                                                                    zIndex: 2,
                                                                }}
                                                            />
                                                        </React.Fragment>
                                                    );
                                                }) : null}

                                        {/* Phase 1 highlight (no anchors) */}
                                        {(!anchors || !anchors[(selectedArtifactId ?? "") as string]) && isHighlighted ? (() => {
                                            const q1 = artifactLookup?.[selectedArtifactId || ""] || [];
                                            const hl2 = computeHighlightRangeFromSpans(pageNumber, q1);
                                            const rx2a = selectedArtifactText ? findRegexBandTwoAnchors(pageNumber, selectedArtifactText) : null;
                                            const rx2b = !rx2a && selectedArtifactText ? findRegexMatchBounds(pageNumber, selectedArtifactText) : null;
                                            const pb2 = selectedArtifactText ? findParagraphBandByPreview(pageNumber, selectedArtifactText) : null;
                                            const band2 = pb2 ?? rx2a ?? rx2b ?? hl2;
                                            return band2 ? (
                                                <div
                                                    aria-hidden
                                                    style={{
                                                        position: "absolute",
                                                        left: Math.max(0, band2.left - 2),
                                                        top: band2.top,
                                                        width: Math.max(8, band2.right - band2.left) + 4,
                                                        height: Math.max(8, band2.bottom - band2.top),
                                                        background: "rgba(250, 204, 21, 0.24)",
                                                        borderRadius: 4,
                                                        pointerEvents: "none",
                                                        zIndex: 1,
                                                    }}
                                                />
                                            ) : null;
                                        })() : null}
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
