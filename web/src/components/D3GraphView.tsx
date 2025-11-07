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

export default function D3GraphView({ graph, height = "70vh", onSelectNode }: Props) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [filterTypes, setFilterTypes] = useState<Record<string, boolean>>({});
    const [query, setQuery] = useState("");
    const [kHop, setKHop] = useState<number>(1);
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [pathEndpoints, setPathEndpoints] = useState<{ a?: string; b?: string }>({});
    const [selectedType, setSelectedType] = useState<string | null>(null);

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
            query || "",
            String(kHop),
            selectedNodeId || "",
            pathEndpoints?.a || "",
            pathEndpoints?.b || "",
            Boolean(onSelectNode) ? "1" : "0",
            String(height),
            selectedType || "",
        ];
        return keyParts.join("|");
    }, [graph?.arxiv_id, graph?.nodes?.length, graph?.edges?.length, filterTypes, query, kHop, selectedNodeId, pathEndpoints?.a, pathEndpoints?.b, onSelectNode, height, selectedType]);

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
            .style("background", "#ffffff");

        // create persistent controls container (so legend doesn't flicker on relayout)
        // Placed at bottom-center and collapsible
        // Render controls outside the SVG container so they don't overlay the graph.
        // Prefer the parent element (the right-hand column) when available.
        const controlsRoot = (containerRef.current && containerRef.current.parentElement) ? containerRef.current.parentElement : containerRef.current;

        // Create (or reuse) a dedicated host element for graph controls so we never create duplicates.
        const graphControlsId = "graph-controls";
        let graphControlsEl = document.getElementById(graphControlsId);
        if (!graphControlsEl) {
            const host = (containerRef.current && containerRef.current.parentElement) ? containerRef.current.parentElement : document.body;
            graphControlsEl = document.createElement("div");
            graphControlsEl.id = graphControlsId;
            graphControlsEl.className = "graph-controls-host";
            // append the host after the graph container to keep document flow
            host.appendChild(graphControlsEl);
        }

        // remove any prev legend inside the host (defensive for HMR)
        graphControlsEl.querySelectorAll("div.__d3_controls").forEach((el) => el.remove());

        // create a single controls block inside the host
        const controlsSel = (d3.select(graphControlsEl) as any)
            .append("div")
            .attr("class", "__d3_controls")
            // render controls as a normal block after the SVG so they do not overlay the viz
            .style("position", "static")
            .style("margin", "8px auto 0 auto")
            .style("max-width", "880px")
            .style("background", "rgba(255,255,255,0.98)")
            .style("padding", "8px")
            .style("border-radius", "10px")
            .style("box-shadow", "0 8px 30px rgba(2,6,23,0.08)")
            .style("font-family", "Inter, sans-serif")
            .style("font-size", "12px")
            .style("max-height", "50vh")
            .style("overflow", "auto")
            .style("pointer-events", "auto");
        const controls = controlsSel.attr("role", "region").attr("aria-label", `Graph controls for ${graph?.arxiv_id || ""}`);

        // Start controls expanded by default so the legend/types are visible.
        // The legend box remains clickable to collapse; this makes the UI discoverable.
        controls.classed("collapsed", false);
        controls.selectAll("*").remove();

        // create legend root (clickable area) and attach toggle handlers
        const legendRoot = controls
            .append("div")
            .style("margin-bottom", "8px")
            // make the legend box visually tappable and large enough to display swatches
            .style("min-height", "40px")
            .style("display", "flex")
            .style("align-items", "center")
            .style("gap", "12px")
            .style("padding", "8px 12px")
            .style("background", "rgba(255,255,255,0.98)")
            .style("border", "1px solid rgba(0,0,0,0.04)")
            .style("border-radius", "8px")
            .attr("role", "button")
            .attr("tabindex", "0")
            .attr("aria-expanded", String(!controls.classed("collapsed")))
            .style("cursor", "pointer")
            .on("click", (event: any) => {
                const collapsed = controls.classed("collapsed");
                controls.classed("collapsed", !collapsed);
                d3.select(event.currentTarget as HTMLElement).attr("aria-expanded", String(!collapsed));
            })
            .on("keydown", (event: any) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    (event.currentTarget as HTMLElement).click();
                }
            });

        // body wrapper for the actual controls (so the header stays visible when collapsed)
        controls.append("div").attr("class", "controls-body").style("margin-top", "8px");

        // Render legend synchronously (outside the async layout) so artifact types are visible immediately.
        // Use nodeTypes derived from the graph to list all types.

        const syncTypes = Array.from(new Set(graph.nodes.map((n) => n.type)));
        const syncTypesBody = controls.select(".controls-body");
        syncTypes.forEach((t: string, i: number) => {
            const row = syncTypesBody
                .append("div")
                .attr("data-type", t)
                .attr("role", "button")
                .attr("tabindex", "0")
                .attr("aria-pressed", "false")
                .style("display", "inline-flex")
                .style("align-items", "center")
                .style("gap", "8px")
                .style("margin", "6px")
                .style("padding", "6px 8px")
                .style("border-radius", "8px")
                .style("cursor", "pointer")
                .style("background", filterTypes[t] ? "rgba(0,0,0,0.02)" : "transparent")
                .on("click", (event: any) => {
                    const cur = selectedType;
                    const nextType = cur === t ? null : t;
                    setSelectedType(nextType);
                    syncTypesBody
                        .selectAll("[data-type]")
                        .each((d: any, i: number, nodes: ArrayLike<HTMLElement>) => {
                            const el = nodes[i] as HTMLElement;
                            const tt = el.getAttribute("data-type") || "";
                            d3.select(el).style("opacity", nextType ? (tt === nextType ? "1" : "0.38") : "1");
                            d3.select(el).style("background", nextType && tt === nextType ? "rgba(0,0,0,0.02)" : "transparent");
                            d3.select(el).attr("aria-pressed", nextType && tt === nextType ? "true" : "false");
                        });
                    d3.select(event.currentTarget).style("opacity", nextType ? (nextType === t ? "1" : "0.38") : "1");
                    d3.select(event.currentTarget as HTMLElement).attr("aria-pressed", nextType ? (nextType === t ? "true" : "false") : "false");
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
                .style("color", "#0f172a");
        });

        // Prepare nodes/links filtered by type and search
        const searchLower = query.trim().toLowerCase();
        const visibleNodes = graph.nodes.filter(
            (n) => filterTypes[n.type] && (!searchLower || (n.display_name || "").toLowerCase().includes(searchLower) || (n.label || "").toLowerCase().includes(searchLower))
        );
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

                // Draw links (paths)
                const linkG = svg.append("g").attr("class", "links");
                const linkEls = linkG
                    .selectAll("path")
                    .data(visibleLinks)
                    .enter()
                    .append("path")
                    .attr("d", (d) => {
                        const s = positions[d.source];
                        const t = positions[d.target];
                        if (!s || !t) return "";
                        const mx = (s.x + t.x) / 2;
                        return `M${s.x},${s.y} C ${mx},${s.y} ${mx},${t.y} ${t.x},${t.y}`;
                    })
                    .attr("stroke", "#8b95a5")
                    .attr("stroke-width", 1.6)
                    .attr("fill", "none")
                    .attr("opacity", 0.9);

                // Node group
                const nodeG = svg.append("g").attr("class", "nodes");

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
                            setSelectedNodeId(d.id);
                            const full = graph.nodes.find((n) => n.id === d.id) || d;
                            onSelectNode?.((full as unknown) as ArtifactNode);
                            const detailsEl = document.querySelector(".artifact-details-root");
                            if (detailsEl) {
                                (detailsEl as HTMLElement).scrollIntoView({ behavior: "smooth", block: "start" });
                            }
                        }
                    });

                node
                    .append("circle")
                    .attr("r", 18)
                    .attr("fill", (d, i) => PASTEL_PALETTE[i % PASTEL_PALETTE.length])
                    .attr("stroke", "#1f2937")
                    .attr("stroke-width", (d: any) => (selectedType ? (d.type === selectedType ? 3 : 1.2) : 1.5))
                    .attr("opacity", (d: any) => (selectedType ? (d.type === selectedType ? 1 : 0.25) : 1));

                node
                    .append("foreignObject")
                    .attr("width", 220)
                    .attr("height", 60)
                    .attr("x", 22)
                    .attr("y", -30)
                    .append("xhtml:div")
                    .style("font-family", "Merriweather, serif")
                    .style("font-weight", "700")
                    .style("font-size", "13px")
                    .style("color", "#0f172a")
                    .style("padding", "2px 6px")
                    .html((d: ArtifactNode) => `<div data-id="${d.id}" style="color:#000 !important;">${d.type}</div>`);

                // Hover & click interactions
                const tooltip = d3
                    .select(containerRef.current)
                    .append("div")
                    .attr("class", "d3-tooltip")
                    .style("position", "absolute")
                    .style("pointer-events", "none")
                    .style("background", "rgba(0,0,0,0.75)")
                    .style("color", "white")
                    .style("padding", "6px 8px")
                    .style("border-radius", "4px")
                    .style("font-size", "12px")
                    .style("display", "none");

                node
                    .on("mouseenter", function (event: MouseEvent, d: ArtifactNode) {
                        tooltip.style("display", "block").html(d.display_name || d.label || "");
                        d3.select(this).select("circle").attr("stroke-width", 2.5);
                    })
                    .on("mousemove", function (event: MouseEvent) {
                        tooltip.style("left", (event as MouseEvent).pageX + 12 + "px").style("top", (event as MouseEvent).pageY + 12 + "px");
                    })
                    .on("mouseleave", function () {
                        tooltip.style("display", "none");
                        d3.select(this).select("circle").attr("stroke-width", 1.5);
                    })
                    .on("click", function (event: MouseEvent, d: ArtifactNode) {
                        setSelectedNodeId(d.id);
                        // Find the full artifact node from the original graph (ensure content_preview is available)
                        const full = graph.nodes.find((n) => n.id === d.id) || d;
                        onSelectNode?.((full as unknown) as ArtifactNode);
                        // scroll details into view for better discoverability
                        const detailsEl = document.querySelector(".artifact-details-root");
                        if (detailsEl) {
                            (detailsEl as HTMLElement).scrollIntoView({ behavior: "smooth", block: "start" });
                        }
                    });

                // k-hop highlighting
                function computeKHop(rootId: string, k: number) {
                    const visited = new Set<string>();
                    let frontier = new Set<string>([rootId]);
                    visited.add(rootId);
                    for (let step = 0; step < k; step++) {
                        const next = new Set<string>();
                        for (const u of frontier) {
                            const neigh = adj[u] || [];
                            for (const v of neigh) {
                                if (!visited.has(v)) {
                                    visited.add(v);
                                    next.add(v);
                                }
                            }
                        }
                        frontier = next;
                        if (frontier.size === 0) break;
                    }
                    return visited;
                }

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

                // apply k-hop or path if requested
                if (selectedNodeId) {
                    const inKHop = computeKHop(selectedNodeId, kHop);
                    node.select("circle").attr("opacity", (d: any) => (inKHop.has(d.id) ? 1 : 0.25));
                    linkEls.attr("opacity", (l: any) => (inKHop.has(l.source) && inKHop.has(l.target) ? 1 : 0.12));
                }

                if (pathEndpoints.a && pathEndpoints.b) {
                    const path = shortestPath(pathEndpoints.a, pathEndpoints.b);
                    if (path && path.length > 0) {
                        const setP = new Set(path);
                        node.select("circle").attr("stroke", (d: any) => (setP.has(d.id) ? "#111827" : "#111827")).attr("stroke-width", (d: any) => (setP.has(d.id) ? 3 : 1.5));
                        linkEls.attr("stroke", (l: any) => (setP.has(l.source) && setP.has(l.target) ? "#1f2937" : "#94a3b8")).attr("stroke-width", (l: any) => (setP.has(l.source) && setP.has(l.target) ? 2.6 : 1.2));
                    }
                }

                // typeset KaTeX in labels
                requestAnimationFrame(() => {
                    const root = containerRef.current;
                    if (root) renderKatex(root).catch(() => { });
                });

                // reuse persistent controls container created outside async — do not remove children here,
                // keep legend and controls already rendered above.
                // controls.selectAll("*").remove();

                // controls-body already created earlier (synchronously) — do not recreate here.
                // (No-op to avoid duplicate controls-body on relayout)
                // const body = controls.select(".controls-body");

                // (Removed the duplicate legend rendering here — we render it synchronously earlier to ensure visibility)

                // search + controls with accessible label and live region
                // hidden label for screen-readers
                controls
                    .append("label")
                    .attr("for", "graph-search")
                    .attr("class", "sr-only")
                    .text("Search artifacts by name or label");

                const inputs = controls.append("div").style("display", "flex").style("flex-direction", "column").style("gap", "6px");
                inputs
                    .append("input")
                    .attr("id", "graph-search")
                    .attr("aria-label", "Search artifacts by name or label")
                    .attr("placeholder", "Search by name or label")
                    .attr("type", "text")
                    .attr("value", query)
                    .on("input", (event: Event) => {
                        setQuery((event.target as HTMLInputElement).value);
                    })
                    .style("width", "100%")
                    .style("padding", "6px")
                    .style("box-sizing", "border-box");

                // aria-live status region intentionally removed per UX request

                const row2 = inputs.append("div").style("display", "flex").style("gap", "6px").style("align-items", "center");
                row2
                    .append("input")
                    .attr("type", "number")
                    .attr("min", 1)
                    .attr("value", kHop.toString())
                    .on("input", (event: any) => setKHop(Math.max(1, Number(event.target.value) || 1)))
                    .style("width", "64px")
                    .style("padding", "6px");

                row2.append("div").text("k-hop").style("font-size", "12px").style("color", "#334155");

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
