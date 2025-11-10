"use client";

import React, { useEffect, useRef, useState, useMemo } from "react";
import * as d3 from "d3";
import type { DocumentGraph, ArtifactNode } from "@/lib/types";
import { unescapeLatex } from "@/lib/latex";
import { renderKatex } from "@/lib/katex";

/**
 * D3 Sugiyama graph viewer with:
 * - layered Sugiyama layout via d3-dag (dynamically imported to avoid typing friction)
 * - pastel, colorblind-safe palette
 * - compact legend, filters, search
 * - k-hop neighborhood highlight and shortest-path highlight
 *
 * For large graphs you may want additional clustering / progressive load.
 */

type Props = {
    graph: DocumentGraph;
    height?: number | string;
    onSelectNode?: (node: ArtifactNode | null) => void;
    selectedNodeId?: string | null;
    showLabels?: boolean;
    zoom?: number;
};

const PASTEL_PALETTE = [
    "#88c0d0",
    "#a3be8c",
    "#ebcb8b",
    "#d08770",
    "#b48ead",
    "#81a1c1",
    "#8fbcbb",
];

export default function D3GraphView({ graph, height = "70vh", onSelectNode, selectedNodeId, showLabels = true, zoom = 1 }: Props) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [filterTypes, setFilterTypes] = useState<Record<string, boolean>>({});
    const [selectedNodeIdState, setSelectedNodeIdState] = useState<string | null>(null);
    const [pathEndpoints, setPathEndpoints] = useState<{ a?: string; b?: string }>({});
    const [selectedType, setSelectedType] = useState<string | null>(null);
    const [legendCollapsed, setLegendCollapsed] = useState<boolean>(false);

    // derive node types once
    const nodeTypes = useMemo(() => {
        const types = Array.from(new Set(graph.nodes.map((n) => n.type)));
        const initial: Record<string, boolean> = {};
        types.forEach((t) => (initial[t] = true));
        return { types, initial };
    }, [graph]);

    useEffect(() => {
        setFilterTypes(nodeTypes.initial);
    }, [nodeTypes]);

    const effectDepsKey = useMemo(() => {
        const keyParts = [
            graph?.arxiv_id || "",
            String(graph?.nodes?.length || 0),
            String(graph?.edges?.length || 0),
            Object.keys(filterTypes || {}).sort().join(","),
            String((selectedNodeId ?? selectedNodeIdState) || ""),
            pathEndpoints?.a || "",
            pathEndpoints?.b || "",
            Boolean(onSelectNode) ? "1" : "0",
            String(height),
            selectedType || "",
            String(showLabels),
            String(zoom),
            legendCollapsed ? "lc1" : "lc0",
        ];
        return keyParts.join("|");
    }, [graph?.arxiv_id, graph?.nodes?.length, graph?.edges?.length, filterTypes, selectedNodeId, pathEndpoints?.a, pathEndpoints?.b, onSelectNode, height, selectedType, showLabels, zoom, legendCollapsed]);

    useEffect(() => {
        if (!containerRef.current) return;
        // clear
        d3.select(containerRef.current).selectAll("*").remove();

        const width = containerRef.current.clientWidth || 1200;
        const h = typeof height === "number" ? height : containerRef.current.clientHeight || 700;

        const svg = d3
            .select(containerRef.current)
            .append("svg")
            .attr("width", "100%")
            .attr("height", h)
            .attr("viewBox", `0 0 ${width} ${h}`)
            .style("background", "var(--surface)");

        // Scene container to enable external zoom via transform
        const sceneG = svg.append("g").attr("class", "scene");

        // Define arrowhead markers for edge direction, sized for visibility
        const defs = svg.append("defs");

        function appendArrow(id: string, fill: string) {
            const m = defs
                .append("marker")
                .attr("id", id)
                .attr("viewBox", "0 0 12 12")
                .attr("refX", 12) // push a bit forward so arrow sits at end
                .attr("refY", 6)
                .attr("markerWidth", 8)
                .attr("markerHeight", 8)
                .attr("orient", "auto")
                .attr("markerUnits", "userSpaceOnUse");
            m.append("path").attr("d", "M 0 0 L 12 6 L 0 12 z").attr("fill", fill);
        }

        // Blue for dependency edges, muted for others
        appendArrow("arrow-dep", "#5561ff");
        appendArrow("arrow-ref", "var(--muted)");
        appendArrow("arrow-gen", "var(--muted)");

        // Bottom legend overlay inside the graph container (no collapsible header; no extra top spacing)
        // Clear any previous overlay (HMR/relayout safety)
        d3.select(containerRef.current).selectAll("div.__legend_overlay").remove();

        const legendOverlay = d3
            .select(containerRef.current)
            .append("div")
            .attr("class", "__legend_overlay")
            .style("position", "absolute")
            .style("left", "0")
            .style("right", "0")
            .style("bottom", "8px")
            .style("display", "flex")
            .style("justify-content", "center")
            .style("pointer-events", "none"); // allow clicking through except legend box

        const legendBox = legendOverlay
            .append("div")
            .style("pointer-events", "auto")
            .style("padding", "8px 12px")
            .style("background", "rgba(255,255,255,0.9)")
            .style("backdrop-filter", "saturate(180%) blur(4px)")
            .style("border", "1px solid rgba(0,0,0,0.06)")
            .style("border-radius", "10px")
            .style("box-shadow", "0 4px 12px rgba(0,0,0,0.06)")
            .style("display", "flex")
            .style("flex-direction", "column")
            .style("gap", "6px")
            .style("max-width", "min(92%, 780px)");

        // Legend header with collapse toggle
        const header = legendBox
            .append("div")
            .style("display", "flex")
            .style("align-items", "center")
            .style("justify-content", "space-between")
            .style("gap", "8px");

        header
            .append("div")
            .text("Legend")
            .style("font-size", "12px")
            .style("font-weight", "600")
            .style("color", "var(--text)");

        header
            .append("button")
            .attr("type", "button")
            .attr("aria-label", legendCollapsed ? "Show legend" : "Hide legend")
            .attr("title", legendCollapsed ? "Show legend" : "Hide legend")
            .style("font-size", "12px")
            .style("padding", "2px 8px")
            .style("border", "1px solid rgba(0,0,0,0.08)")
            .style("border-radius", "6px")
            .style("background", "rgba(0,0,0,0.03)")
            .style("cursor", "pointer")
            .html(
                legendCollapsed
                    ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                         <path d="M6 9l6 6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                       </svg>`
                    : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                         <path d="M6 15l6-6 6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                       </svg>`
            )
            .on("click", () => setLegendCollapsed((v) => !v));

        const content = legendBox
            .append("div")
            .attr("class", "__legend_content")
            .style("display", "flex")
            .style("flex-wrap", "wrap")
            .style("gap", "8px 12px")
            .style("align-items", "center")
            .style("overflow", "hidden")
            .style("max-height", legendCollapsed ? "0px" : "300px")
            .style("transition", "max-height 200ms ease");

        const syncTypes = Array.from(new Set(graph.nodes.map((n) => n.type)));
        syncTypes.forEach((t: string, i: number) => {
            const row = content
                .append("div")
                .attr("data-type", t)
                .attr("role", "button")
                .attr("tabindex", "0")
                .attr("aria-pressed", selectedType === t ? "true" : "false")
                .style("display", "inline-flex")
                .style("align-items", "center")
                .style("gap", "8px")
                .style("padding", "6px 8px")
                .style("border-radius", "8px")
                .style("cursor", "pointer")
                .style("background", selectedType ? (selectedType === t ? "rgba(0,0,0,0.03)" : "transparent") : "transparent")
                .style("opacity", selectedType ? (selectedType === t ? "1" : "0.5") : "1")
                .on("click", (event: any) => {
                    const cur = selectedType;
                    const nextType = cur === t ? null : t;
                    setSelectedType(nextType);
                    // update visual press/opacity (use this-typed callback to satisfy d3 typings)
                    legendBox.selectAll("[data-type]").each(function () {
                        const el = this as HTMLElement;
                        const tt = el.getAttribute("data-type") || "";
                        d3.select(el)
                            .attr("aria-pressed", nextType && tt === nextType ? "true" : "false")
                            .style("background", nextType && tt === nextType ? "rgba(0,0,0,0.03)" : "transparent")
                            .style("opacity", nextType ? (tt === nextType ? "1" : "0.5") : "1");
                    });
                })
                .on("keydown", (event: any) => {
                    if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        (event.currentTarget as HTMLElement).click();
                    }
                });

            row
                .append("div")
                .style("width", "14px")
                .style("height", "14px")
                .style("background", PASTEL_PALETTE[i % PASTEL_PALETTE.length])
                .style("border-radius", "4px")
                .style("flex", "0 0 auto");

            row
                .append("div")
                .text(t)
                .style("font-size", "12px")
                .style("color", "var(--text)");
        });

        // Prepare nodes/links filtered by type and search
        const visibleNodes = graph.nodes.filter((n) => filterTypes[n.type]);
        const visibleIds = new Set(visibleNodes.map((n) => n.id));
        const visibleLinks = graph.edges.filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target));

        // adjacency for BFS / path
        const adj: Record<string, string[]> = {};
        graph.nodes.forEach((n) => (adj[n.id] = []));
        graph.edges.forEach((e) => {
            adj[e.source] = adj[e.source] || [];
            adj[e.source].push(e.target);
            // keep undirected access for path/k-hop
            adj[e.target] = adj[e.target] || [];
            adj[e.target].push(e.source);
        });

        // Build dag nodes for d3-dag (id + children)
        const childrenMap: Record<string, string[]> = {};
        visibleNodes.forEach((n) => (childrenMap[n.id] = []));
        visibleLinks.forEach((l) => {
            childrenMap[l.source].push(l.target);
        });
        const dagData = Object.entries(childrenMap).map(([id, children]) => ({ id, children }));

        // Dynamically import d3-dag to avoid TS typing friction
        (async () => {
            try {
                const daglib = await import("d3-dag");
                const { graphStratify, sugiyama, layeringSimplex, decrossTwoLayer } = daglib as any;

                // Build DAG
                let dag;
                try {
                    dag = graphStratify()(dagData);
                } catch {
                    // fallback: create simple fake nodes of a single layer if stratify fails
                    dag = { descendants: () => [] };
                }

                // Run sugiyama
                const layout = (sugiyama as any)()
                    .layering(layeringSimplex())
                    .decross(decrossTwoLayer())
                    .nodeSize([140, 90]);

                try {
                    layout(dag);
                } catch {
                    // If layout fails, continue with no dag coords
                }

                // Compute positions: use dag coords when available else fallback to grid
                const descendants: any[] = (dag && dag.descendants && dag.descendants()) || [];
                const positions: Record<string, { x: number; y: number }> = {};
                if (descendants.length > 0) {
                    const xs = descendants.map((d) => d.x).filter((v: any) => typeof v === "number");
                    const ys = descendants.map((d) => d.y).filter((v: any) => typeof v === "number");
                    const minX = Math.min(...xs);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);
                    const maxY = Math.max(...ys);

                    const xScale = d3.scaleLinear().domain([minX, maxX]).range([80, width - 80]);
                    const yScale = d3.scaleLinear().domain([minY, maxY]).range([60, h - 120]);

                    descendants.forEach((d) => {
                        positions[d.id] = { x: xScale(d.x), y: yScale(d.y) };
                    });
                } else {
                    // simple circle layout
                    const radius = Math.min(width, h) / 3;
                    const centerX = width / 2;
                    const centerY = h / 2;
                    visibleNodes.forEach((n, i) => {
                        const angle = (i / visibleNodes.length) * Math.PI * 2;
                        positions[n.id] = { x: centerX + radius * Math.cos(angle), y: centerY + radius * Math.sin(angle) };
                    });
                }

                // Draw links (paths) with direction and styling by type
                const linkG = sceneG.append("g").attr("class", "links");
                // Stabilize path IDs so dependency labels attach to the correct edge even after filtering
                (visibleLinks as any[]).forEach((l, i) => ((l as any).__edgeId = `edge-${i}`));
                const linkEls = linkG
                    .selectAll("path")
                    .data(visibleLinks)
                    .enter()
                    .append("path")
                    .attr("id", (d: any) => d.__edgeId)
                    .attr("class", (d: any) => {
                        const kind = d.dependency_type ? "dep" : (d.reference_type ? "ref" : "gen");
                        return `link ${kind}`;
                    })
                    .attr("d", (d) => {
                        const s = positions[d.source];
                        const t = positions[d.target];
                        if (!s || !t) return "";
                        // Offset start/end so the path touches the node circle instead of its center
                        // Draw a straight line so the arrowhead is perfectly aligned between centers.
                        const nodeR = 18;            // circle radius
                        const pad = 2;               // small pad so arrow tip is just outside the circle
                        const dx = t.x - s.x;
                        const dy = t.y - s.y;
                        const len = Math.max(1e-6, Math.hypot(dx, dy));
                        const ux = dx / len;
                        const uy = dy / len;

                        // start just outside the source circle
                        const sx = s.x + ux * (nodeR + pad);
                        const sy = s.y + uy * (nodeR + pad);
                        // end just outside the target circle; marker tip sits at the path end
                        const tx = t.x - ux * (nodeR + pad);
                        const ty = t.y - uy * (nodeR + pad);

                        return `M${sx},${sy} L ${tx},${ty}`;
                    })
                    .attr("stroke", (d: any) => {
                        // Emphasize dependency edges; keep references muted
                        if (d.dependency_type) return "#5561ff"; // primary for deps
                        if (d.reference_type) return "var(--muted)";
                        return "var(--muted)";
                    })
                    // All edges full (no dashes)
                    .attr("stroke-dasharray", null)
                    .attr("stroke-width", (d: any) => (d.dependency_type ? 2.2 : 1.6))
                    .attr("fill", "none")
                    .attr("opacity", 0.95)
                    .attr("marker-end", (d: any) => {
                        if (d.dependency_type) return "url(#arrow-dep)";
                        if (d.reference_type) return "url(#arrow-ref)";
                        return "url(#arrow-gen)";
                    });

                // Edge tooltip shown on hover (instead of writing on the edge)
                const edgeTip = d3
                    .select(containerRef.current)
                    .append("div")
                    .attr("class", "d3-edge-tooltip")
                    .style("position", "absolute")
                    .style("pointer-events", "none")
                    .style("display", "none")
                    .style("padding", "4px 6px")
                    .style("font-size", "12px")
                    .style("background", "rgba(255,255,255,0.95)")
                    .style("border", "1px solid rgba(0,0,0,0.1)")
                    .style("border-radius", "6px")
                    .style("box-shadow", "0 2px 6px rgba(0,0,0,0.08)")
                    .style("color", "#111827"); // slate-900

                const nameMap: Record<string, string> = {};
                graph.nodes.forEach((n) => {
                    nameMap[n.id] = n.display_name || n.label || n.id;
                });

                let pinnedEdge: SVGPathElement | null = null;

                function showEdgeTipForPath(el: SVGPathElement, d: any) {
                    // midpoint of the path
                    const len = el.getTotalLength();
                    const mid = el.getPointAtLength(len / 2);
                    // position relative to container
                    const container = containerRef.current!;
                    const rect = (container.firstElementChild as SVGSVGElement).getBoundingClientRect();
                    // container is relatively positioned; we can place absolutely within
                    edgeTip
                        .style("left", `${mid.x + 8}px`)
                        .style("top", `${mid.y + 8}px`)
                        .style("display", "block")
                        .html(() => {
                            const kind = (d.dependency_type || d.type || "").toString().replace(/_/g, " ");
                            const src = nameMap[d.source] || d.source;
                            const tgt = nameMap[d.target] || d.target;
                            const just = d.dependency || "";
                            const badge = kind ? `<span style="font-weight:600;color:#374151">${kind}</span>` : `<span style="color:#6b7280">reference</span>`;
                            const arrow = "→";
                            const ctx = just ? `<div style="margin-top:2px;color:#4b5563">${just}</div>` : "";
                            return `${badge}<div style="color:#1f2937">${src} ${arrow} ${tgt}</div>${ctx}`;
                        });
                }

                linkEls
                    .on("mouseenter", function (_ev: MouseEvent, d: any) {
                        d3.select(this).attr("stroke-width", (d: any) => (d.dependency_type ? 3 : 2));
                        showEdgeTipForPath(this as SVGPathElement, d);
                    })
                    .on("mousemove", function (_ev: MouseEvent, d: any) {
                        // follow cursor slightly by recomputing midpoint (keeps label near edge)
                        showEdgeTipForPath(this as SVGPathElement, d);
                    })
                    .on("mouseleave", function () {
                        d3.select(this).attr("stroke-width", (d: any) => (d.dependency_type ? 2.2 : 1.6));
                        if (pinnedEdge !== this) {
                            edgeTip.style("display", "none");
                        }
                    })
                    .on("click", function (_ev: MouseEvent, d: any) {
                        // Toggle pinning
                        if (pinnedEdge === this) {
                            pinnedEdge = null;
                            edgeTip.style("display", "none");
                        } else {
                            pinnedEdge = this as SVGPathElement;
                            showEdgeTipForPath(this as SVGPathElement, d);
                        }
                    });

                // Clicking empty space unpins
                svg.on("click", function (ev: any) {
                    // Ignore if the target is a path (handled above)
                    if (ev && ev.target && ev.target.tagName === "path") return;
                    pinnedEdge = null;
                    edgeTip.style("display", "none");
                });

                // Node group
                const nodeG = sceneG.append("g").attr("class", "nodes");

                const node = nodeG
                    .selectAll("g.node")
                    .data(visibleNodes)
                    .enter()
                    .append("g")
                    .attr("class", "node")
                    .attr("transform", (d) => `translate(${positions[d.id].x},${positions[d.id].y})`)
                    .attr("role", "button")
                    .attr("tabindex", "0")
                    .attr("aria-label", (d: ArtifactNode) => d.type || d.display_name || d.label || d.id)
                    .style("cursor", "pointer")
                    .on("keydown", function (event: KeyboardEvent, d: ArtifactNode) {
                        if ((event as KeyboardEvent).key === "Enter" || (event as KeyboardEvent).key === " ") {
                            (event as KeyboardEvent).preventDefault();
                            // emulate click behaviour
                            setSelectedNodeIdState(d.id);
                            const full = graph.nodes.find((n) => n.id === d.id) || d;
                            onSelectNode?.((full as unknown) as ArtifactNode);
                            // Do not scroll the page to details; keep reader centered behavior only.
                        }
                    });

                node
                    .append("circle")
                    .attr("r", 18)
                    .attr("fill", (d, i) => PASTEL_PALETTE[i % PASTEL_PALETTE.length])
                    .attr("stroke", "var(--text)")
                    .attr("stroke-width", (d: any) => {
                        if (selectedType) return d.type === selectedType ? 3 : 1.2;
                        const selId = (selectedNodeId ?? selectedNodeIdState) || null;
                        return selId && d.id === selId ? 3.2 : 1.5;
                    })
                    .attr("opacity", (d: any) => (selectedType ? (d.type === selectedType ? 1 : 0.25) : 1));

                if (showLabels) {
                    node
                        .append("foreignObject")
                        .attr("width", 220)
                        .attr("height", 60)
                        .attr("x", 22)
                        .attr("y", -30)
                        .append("xhtml:div")
                        .html((d: ArtifactNode) => {
                            const name = d.display_name || d.label || d.id;
                            return `<div data-id="${d.id}" style="font-size:12px;color:var(--text)">${name}</div>`;
                        });
                }

                // Hover & click interactions
                // Node hover tooltip disabled per UX: no floating text on hover.

                node
                    .on("mouseenter", function () {
                        d3.select(this).select("circle").attr("stroke-width", 2.5);
                    })
                    .on("mousemove", null)
                    .on("mouseleave", function () {
                        d3.select(this).select("circle").attr("stroke-width", 1.5);
                    })
                    .on("click", function (event: MouseEvent, d: ArtifactNode) {
                        setSelectedNodeIdState(d.id);
                        // Find the full artifact node from the original graph (ensure content_preview is available)
                        const full = graph.nodes.find((n) => n.id === d.id) || d;
                        onSelectNode?.((full as unknown) as ArtifactNode);
                        // Do not scroll the page to details; keep reader centered behavior only.
                    });


                // shortest path (BFS)
                function shortestPath(a: string, b: string) {
                    if (a === b) return [a];
                    const q: string[] = [a];
                    const parent: Record<string, string | null> = {};
                    parent[a] = null;
                    const seen = new Set<string>([a]);
                    while (q.length) {
                        const u = q.shift()!;
                        const neigh = adj[u] || [];
                        for (const v of neigh) {
                            if (!seen.has(v)) {
                                parent[v] = u;
                                if (v === b) {
                                    // reconstruct
                                    const path = [];
                                    let cur: string | null = b;
                                    while (cur) {
                                        path.push(cur);
                                        cur = parent[cur] ?? null;
                                    }
                                    return path.reverse();
                                }
                                seen.add(v);
                                q.push(v);
                            }
                        }
                    }
                    return null;
                }

                // k-hop highlighting removed per UX request

                if (pathEndpoints.a && pathEndpoints.b) {
                    const path = shortestPath(pathEndpoints.a, pathEndpoints.b);
                    if (path && path.length > 0) {
                        const setP = new Set(path);
                        node.select("circle").attr("stroke", (d: any) => (setP.has(d.id) ? "var(--text)" : "var(--text)")).attr("stroke-width", (d: any) => (setP.has(d.id) ? 3 : 1.5));
                        linkEls.attr("stroke", (l: any) => (setP.has(l.source) && setP.has(l.target) ? "var(--text)" : "var(--muted)")).attr("stroke-width", (l: any) => (setP.has(l.source) && setP.has(l.target) ? 2.6 : 1.2));
                    }
                }

                // typeset KaTeX in labels
                requestAnimationFrame(() => {
                    const root = containerRef.current;
                    if (root) renderKatex(root).catch(() => { });
                });

                // Apply external zoom scaling around the center
                try {
                    const zx = width / 2;
                    const zy = h / 2;
                    (sceneG as any).attr("transform", `translate(${zx},${zy}) scale(${zoom}) translate(${-zx},${-zy})`);
                } catch { }

                // reuse persistent controls container created outside async — do not remove children here,
                // keep legend and controls already rendered above.
                // controls.selectAll("*").remove();

                // controls-body already created earlier (synchronously) — do not recreate here.
                // (No-op to avoid duplicate controls-body on relayout)
                // const body = controls.select(".controls-body");

                // (Removed the duplicate legend rendering here — we render it synchronously earlier to ensure visibility)

                // search + controls with accessible label and live region
                // hidden label for screen-readers

                // shortest-path inputs removed from legend per UX request
                // (path endpoints can still be set programmatically or via other UI if desired)

                // Removed path & center-selection buttons per UX request.
                // Path endpoints and centering may be handled via alternate UI if reintroduced later.

                // keyboard help removed per UX request (keep UI minimal)

                // cleanup
                return () => {
                    d3.select(containerRef.current).selectAll("*").remove();
                };
            } catch (err) {
                // d3-dag import/layout failed; leave a simple fallback render (force was previous)
                console.warn("d3-dag failed, skipping Sugiyama layout", err);
                d3.select(containerRef.current).append("div").text("Graph layout error");
            }
        })();

        // cleanup on effect re-run
        return () => {
            d3.select(containerRef.current).selectAll("*").remove();
        };
        // Use a stable dependency list made of primitives so React's dependency array length stays constant.
        // We include counts and specific fields rather than whole objects.
    }, [effectDepsKey]);

    return <div ref={containerRef} style={{ width: "100%", height: typeof height === "number" ? `${height}px` : height, position: "relative" }} />;
}
