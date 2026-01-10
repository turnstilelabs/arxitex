'use client';

import { drag as d3Drag } from 'd3-drag';
import { forceCollide, forceLink, forceManyBody, forceSimulation } from 'd3-force';
import { hierarchy, pack } from 'd3-hierarchy';
import { interpolateLab } from 'd3-interpolate';
import { scaleLinear, scaleSqrt } from 'd3-scale';
import { select } from 'd3-selection';
import { extent } from 'd3-array';
import { zoom as d3Zoom, zoomIdentity } from 'd3-zoom';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';

const ARROW = '→';

type ComponentJson = {
    rank: number;
    stats: {
        node_count: number;
        edge_count: number;
        top_out_degree: Array<[string, number]>;
    };
    nodes: string[];
    edges: Array<{ source: string; target: string }>;
};

type PaperTitlesJson = Record<string, string>;

type PaperNode = {
    id: string;
    componentRank: number;
    artifactCount: number | null;
    inDegree: number;
    outDegree: number;
    // d3 simulation fields
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    fx?: number | null;
    fy?: number | null;
    r: number;
};

type CitationLink = {
    source: string | PaperNode;
    target: string | PaperNode;
};

const DEFAULT_TOP_K = 5;

// Initial warmup ticks (small, fast) to avoid a totally chaotic first paint.
// Remaining ticks are run in requestAnimationFrame batches so we don't block the main thread.
const INITIAL_WARMUP_TICKS = 24;
const TOTAL_LAYOUT_TICKS = 220;
const RAF_TICK_BATCH = 12;

function useResizeObserver<T extends HTMLElement>() {
    const ref = useRef<T | null>(null);
    const [rect, setRect] = useState<{ width: number; height: number } | null>(null);

    useEffect(() => {
        const el = ref.current;
        if (!el) return;

        const ro = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            const cr = entry.contentRect;
            setRect({ width: cr.width, height: cr.height });
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    return { ref, rect };
}

function clusterForce(
    nodes: PaperNode[],
    centersByRank: Map<number, { x: number; y: number; r: number }>,
    strength = 0.08,
) {
    // d3 custom force signature
    function force(alpha: number) {
        for (const n of nodes) {
            const c = centersByRank.get(n.componentRank);
            if (!c || typeof n.x !== 'number' || typeof n.y !== 'number') continue;
            n.vx = (n.vx ?? 0) + (c.x - n.x) * strength * alpha;
            n.vy = (n.vy ?? 0) + (c.y - n.y) * strength * alpha;
        }
    }
    (force as any).initialize = () => {
        // no-op
    };
    return force as any;
}

function containInIslandsForce(
    nodes: PaperNode[],
    centersByRank: Map<number, { x: number; y: number; r: number }>,
    padding = 10,
) {
    function force() {
        for (const n of nodes) {
            const c = centersByRank.get(n.componentRank);
            if (!c || typeof n.x !== 'number' || typeof n.y !== 'number') continue;

            const dx = n.x - c.x;
            const dy = n.y - c.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const max = Math.max(1, c.r - padding - n.r);
            if (dist <= max) continue;

            const k = max / dist;
            n.x = c.x + dx * k;
            n.y = c.y + dy * k;
            n.vx = (n.vx ?? 0) * 0.6;
            n.vy = (n.vy ?? 0) * 0.6;
        }
    }
    (force as any).initialize = () => {
        // no-op
    };
    return force as any;
}

export default function HomeCitationIslands({
    topK = DEFAULT_TOP_K,
    fullscreen = false,
}: {
    topK?: number;
    fullscreen?: boolean;
}) {
    const router = useRouter();
    const { ref: containerRef, rect } = useResizeObserver<HTMLDivElement>();
    const svgRef = useRef<SVGSVGElement | null>(null);


    const [loading, setLoading] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [components, setComponents] = useState<ComponentJson[]>([]);
    const [artifactCounts, setArtifactCounts] = useState<Record<string, number>>({});
    const [paperTitles, setPaperTitles] = useState<PaperTitlesJson>({});

    const rootRef = useRef<HTMLDivElement | null>(null);

    // Lightweight tooltip (absolute positioned)
    const tooltipRef = useRef<HTMLDivElement | null>(null);

    const rafIdRef = useRef<number | null>(null);


    useEffect(() => {
        let cancelled = false;
        async function load() {
            setLoading(true);
            setLoadError(null);
            try {
                const [countsRes, titlesRes, ...compRes] = await Promise.all([
                    fetch('/extref_components/artifact_counts.json', { cache: 'force-cache' }),
                    fetch('/extref_components/paper_titles.json', { cache: 'force-cache' }),
                    ...Array.from({ length: topK }, (_, i) => {
                        const rank = String(i + 1).padStart(3, '0');
                        return fetch(`/extref_components/component_${rank}.json`, { cache: 'force-cache' });
                    }),
                ]);

                if (!countsRes.ok) throw new Error(`Failed to load artifact_counts.json (${countsRes.status})`);
                const counts = (await countsRes.json()) as Record<string, number>;

                // Titles are optional; if missing, we fall back to IDs.
                const titles: PaperTitlesJson = titlesRes.ok ? ((await titlesRes.json()) as PaperTitlesJson) : {};

                const comps: ComponentJson[] = [];
                for (const r of compRes) {
                    if (!r.ok) throw new Error(`Failed to load component JSON (${r.status})`);
                    comps.push((await r.json()) as ComponentJson);
                }

                if (cancelled) return;
                setArtifactCounts(counts);
                setPaperTitles(titles);
                setComponents(comps);
            } catch (e: any) {
                if (cancelled) return;
                setLoadError(e?.message ?? String(e));
            } finally {
                if (cancelled) return;
                setLoading(false);
            }
        }
        void load();
        return () => {
            cancelled = true;
        };
    }, [topK]);

    const graph = useMemo(() => {
        if (!components.length) return null;

        const nodes: PaperNode[] = [];
        const links: CitationLink[] = [];

        // Degrees per paper (within component)
        const degIn = new Map<string, number>();
        const degOut = new Map<string, number>();

        for (const comp of components) {
            for (const e of comp.edges) {
                degOut.set(e.source, (degOut.get(e.source) ?? 0) + 1);
                degIn.set(e.target, (degIn.get(e.target) ?? 0) + 1);
            }
        }

        // Artifact counts might be absent for many papers (not processed); keep as null.
        // For sizing, treat unknown as 0 but style it differently in the UI.
        const artifactValues = components.flatMap((c) => c.nodes.map((id) => artifactCounts[id] ?? 0));
        const maxArtifacts = Math.max(1, ...artifactValues);
        // Make nodes noticeably larger to highlight structure.
        const sizeScale = scaleSqrt().domain([0, maxArtifacts]).range([6, 18]);

        for (const comp of components) {
            for (const id of comp.nodes) {
                const ac = typeof artifactCounts[id] === 'number' ? artifactCounts[id] : null;
                nodes.push({
                    id,
                    componentRank: comp.rank,
                    artifactCount: ac,
                    inDegree: degIn.get(id) ?? 0,
                    outDegree: degOut.get(id) ?? 0,
                    r: sizeScale(ac ?? 0),
                });
            }
            for (const e of comp.edges) {
                links.push({ source: e.source, target: e.target });
            }
        }

        return { nodes, links };
    }, [components, artifactCounts]);


    // Render / update graph
    useEffect(() => {
        const svgEl = svgRef.current;
        if (!svgEl) return;
        if (!rect) return;
        if (!graph) return;

        const rootEl = rootRef.current;
        if (!rootEl) return;

        const width = Math.max(1, rect.width);
        const height = Math.max(320, rect.height);

        const svg = select(svgEl);
        svg.selectAll('*').remove();
        svg.attr('viewBox', `0 0 ${width} ${height}`);

        // Root group for the viz.
        // We make it visible quickly (after a short warmup) to improve perceived load time,
        // and then refine the layout in RAF chunks.
        const g = svg.append('g').attr('opacity', 0);

        // Allow panning/zooming to explore the full-bleed graph.
        // NOTE: we do not show UI controls; mousewheel/trackpad zoom + drag-pan.
        const zoom = d3Zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.2, 6])
            .on('zoom', (event) => {
                g.attr('transform', event.transform.toString());
            });

        svg.call(zoom as any);

        // Arrow marker (directed citations)
        svg
            .append('defs')
            .append('marker')
            .attr('id', 'citation-arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 12)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', 'rgba(170,170,170,0.75)');

        // Layout islands using circle packing
        const packRoot = hierarchy({
            name: 'root',
            children: components.map((c) => ({
                name: `#${String(c.rank).padStart(2, '0')}`,
                rank: c.rank,
                value: c.stats.node_count,
                node_count: c.stats.node_count,
                edge_count: c.stats.edge_count,
            })),
        } as any)
            .sum((d: any) => d.value || 0);

        // Use packing to position island *centers*, then scale radii up so
        // each island has more room and the graph fills the viewport.
        const packLayout = pack<any>().size([width, height]).padding(6);
        const packed = packLayout(packRoot);
        const leaves = packed.leaves();

        const centersByRank = new Map<number, { x: number; y: number; r: number; nodeCount: number; edgeCount: number }>();
        const islandRadiusScale = 1.8;
        const cx = width / 2;
        const cy = height / 2;

        // Use the circle-pack layout directly for component centers.
        // This fills the rectangle much better than putting components on a ring (which creates a big hole in the middle).
        // We also slightly push centers outward to reduce unused space near the edges.
        const spread = 1.12;

        for (const l of leaves) {
            const rank = Number(l.data.rank);
            const r = l.r * islandRadiusScale;

            let x = cx + (l.x - cx) * spread;
            let y = cy + (l.y - cy) * spread;

            // Clamp so we don't push centers beyond the viewport.
            x = Math.max(r, Math.min(width - r, x));
            y = Math.max(r, Math.min(height - r, y));

            centersByRank.set(rank, {
                x,
                y,
                r,
                nodeCount: Number(l.data.node_count) || 0,
                edgeCount: Number(l.data.edge_count) || 0,
            });
        }

        // NOTE: we intentionally do NOT render component-level (island) circles or labels on the homepage.
        // We keep centersByRank for the simulation so papers still cluster by component.

        // Seed node positions near their island center (wider initial spread)
        for (const n of graph.nodes) {
            const c = centersByRank.get(n.componentRank);
            const angle = Math.random() * Math.PI * 2;
            const radius = (c?.r ?? 50) * 0.62 * Math.sqrt(Math.random());
            n.x = (c?.x ?? width / 2) + Math.cos(angle) * radius;
            n.y = (c?.y ?? height / 2) + Math.sin(angle) * radius;
        }

        const linkG = g.append('g').attr('class', 'citation-links');
        const nodeG = g.append('g').attr('class', 'citation-nodes');

        const link = linkG
            .selectAll('path')
            .data(graph.links)
            .enter()
            .append('path')
            .attr('stroke', 'rgba(170,170,170,0.55)')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '2,4')
            .attr('fill', 'none')
            .attr('marker-end', 'url(#citation-arrow)');

        // Color scale: small nodes lighter yellow, big nodes richer yellow.
        const rExtent = extent(graph.nodes.map((n) => n.r)) as [number, number];
        const colorT = scaleLinear().domain([rExtent[0] ?? 0, rExtent[1] ?? 1]).range([0, 1]);
        const nodeFill = (d: PaperNode) => {
            // d3.interpolateLab gives a pleasant perceptual gradient.
            const t = colorT(d.r);
            return interpolateLab('#FFEFA3', '#FFD74A')(t);
        };

        const node = nodeG
            .selectAll('circle')
            .data(graph.nodes)
            .enter()
            .append('circle')
            .attr('r', (d) => d.r)
            .attr('fill', (d) => (d.artifactCount == null ? 'rgba(255,239,163,0.22)' : nodeFill(d)))
            .attr('stroke', 'rgba(18,18,18,0.75)')
            .attr('stroke-width', 1.2)
            .style('pointer-events', 'none');

        const showTooltip = (event: any, d: PaperNode) => {
            const tt = tooltipRef.current;
            if (!tt) return;
            tt.style.opacity = '1';
            const bounds = rootEl.getBoundingClientRect();
            tt.style.left = `${bounds.left + (event.offsetX ?? 0) + 12}px`;
            tt.style.top = `${bounds.top + (event.offsetY ?? 0) + 12}px`;
            const artifactLabel = d.artifactCount == null ? 'unknown' : String(d.artifactCount);
            const title = paperTitles[d.id] ?? d.id;
            tt.innerHTML = `
          <div style="font-weight:700; color: var(--primary-text)">${title}</div>
          <div style="color: var(--secondary-text); margin-top: 2px; font-size: 12px">${d.id}</div>
          <div style="color: var(--secondary-text); margin-top: 2px">
            ${artifactLabel} artifacts · ${d.outDegree} cites · ${d.inDegree} cited-by
          </div>
          <div style="color: var(--secondary-text); margin-top: 6px; font-size: 12px">
            Click to open artifact dependency graph →
          </div>
        `;
        };

        // A larger invisible hit-target to make clicking/hovering easier.
        const nodeHit = nodeG
            .selectAll('circle.hit')
            .data(graph.nodes)
            .enter()
            .append('circle')
            .attr('class', 'hit')
            .attr('r', (d) => d.r + 8)
            .attr('fill', 'transparent')
            .style('cursor', 'pointer')
            .style('touch-action', 'none')
            .style('pointer-events', 'all')
            .on('click', (event, d) => {
                showTooltip(event, d);
            })
            .on('dblclick', (_, d) => {
                router.push(`/paper/${encodeURIComponent(d.id)}`);
            })
            .on('mousemove', (event, d) => {
                showTooltip(event, d);
            })
            .on('mouseleave', () => {
                const tt = tooltipRef.current;
                if (!tt) return;
                tt.style.opacity = '0';
            });

        function linkPath(d: any) {
            const s = d.source as PaperNode;
            const t = d.target as PaperNode;
            if (!s || !t || typeof s.x !== 'number' || typeof s.y !== 'number' || typeof t.x !== 'number' || typeof t.y !== 'number') {
                return '';
            }
            // Straight path; keep marker visible by shortening a bit at the target end.
            const dx = t.x - s.x;
            const dy = t.y - s.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;

            const padS = (s.r ?? 0) + 1;
            const padT = (t.r ?? 0) + 8; // extra room for arrowhead
            const x1 = s.x + (dx / dist) * padS;
            const y1 = s.y + (dy / dist) * padS;
            const x2 = t.x - (dx / dist) * padT;
            const y2 = t.y - (dy / dist) * padT;

            return `M${x1},${y1}L${x2},${y2}`;
        }

        const updatePositions = () => {
            node.attr('cx', (d) => d.x ?? 0).attr('cy', (d) => d.y ?? 0);
            nodeHit.attr('cx', (d) => d.x ?? 0).attr('cy', (d) => d.y ?? 0);
            link.attr('d', linkPath);
        };

        // Fit the whole layout to the viewport and fade it in.
        // We run a fixed number of simulation ticks *off-screen* before first paint,
        // to avoid the visible “glitch” where the graph animates then snaps/resizes.
        const fitToViewport = () => {
            const xs: number[] = [];
            const ys: number[] = [];
            for (const n of graph.nodes) {
                if (typeof n.x === 'number' && typeof n.y === 'number') {
                    xs.push(n.x);
                    ys.push(n.y);
                }
            }
            if (!xs.length) return;

            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);

            const bboxW = Math.max(1, maxX - minX);
            const bboxH = Math.max(1, maxY - minY);
            // Smaller pad + multiplier => more zoomed-in.
            // On the homepage background, we want the graph to "touch" the viewport
            // much more aggressively (nodes in all corners).
            const pad = fullscreen ? 0 : 16;
            const zoomBoost = fullscreen ? 1.75 : 1.15;

            const base = Math.min(width / (bboxW + pad), height / (bboxH + pad));
            const k = Math.max(0.35, Math.min(4.0, base * zoomBoost));

            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const x = width / 2 - centerX * k;
            const y = height / 2 - centerY * k;

            const t = zoomIdentity.translate(x, y).scale(k);
            svg.call(zoom.transform as any, t);

            // Reveal only after we have a stable transform.
            g.attr('opacity', 1);
        };

        // Simulation
        const sim = forceSimulation<PaperNode>(graph.nodes)
            .force(
                'link',
                forceLink<PaperNode, any>(graph.links as any)
                    .id((d: any) => d.id)
                    .distance(92)
                    .strength(0.055),
            )
            .force('charge', forceManyBody().strength(-165))
            .force('collide', forceCollide<PaperNode>().radius((d) => d.r + 12).iterations(3))
            // Softer component clustering so nodes can use more space, but still stay broadly grouped.
            .force('cluster', clusterForce(graph.nodes, centersByRank as any, 0.06));

        // We used to run all ticks synchronously before first paint.
        // That looks stable but creates a visible stall on page load.
        // New approach:
        // 1) Run a small warmup synchronously.
        // 2) Paint + fit to viewport.
        // 3) Finish the remaining ticks in RAF batches to keep the UI responsive.
        sim.alpha(1).stop();

        const warmup = Math.min(INITIAL_WARMUP_TICKS, TOTAL_LAYOUT_TICKS);
        sim.tick(warmup);
        updatePositions();
        fitToViewport();

        let ticksDone = warmup;
        const tickInRaf = () => {
            const remaining = TOTAL_LAYOUT_TICKS - ticksDone;
            if (remaining <= 0) return;

            const batch = Math.min(RAF_TICK_BATCH, remaining);
            sim.tick(batch);
            ticksDone += batch;
            updatePositions();

            rafIdRef.current = requestAnimationFrame(tickInRaf);
        };
        rafIdRef.current = requestAnimationFrame(tickInRaf);

        // After initial stabilization, keep the simulation available for interaction.
        // It will only run when users drag nodes.
        sim.on('tick', updatePositions);
        sim.alpha(0);

        // Drag nodes (single click + drag). During drag, reheat the simulation so
        // connected nodes/edges respond (like the paper artifact graph).
        const drag = d3Drag<SVGCircleElement, PaperNode>()
            .on('start', (event, d) => {
                event.sourceEvent?.stopPropagation?.();
                // Reheat so the whole graph responds.
                if (!event.active) sim.alphaTarget(0.25).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                event.sourceEvent?.stopPropagation?.();
                // Release the node after drag; graph will settle.
                d.fx = null;
                d.fy = null;
                if (!event.active) sim.alphaTarget(0);
            });

        nodeHit.call(drag as any);

        return () => {
            if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);
            sim.stop();
        };
    }, [graph, rect, components, paperTitles, router, fullscreen]);

    const graphEl = (
        <div
            ref={(el) => {
                (rootRef as any).current = el;
                (containerRef as any).current = el;
            }}
            className={fullscreen ? 'relative w-full h-full' : 'relative mt-4 w-full'}
            style={
                fullscreen
                    ? {
                        width: '100%',
                        height: '100%',
                        background: 'var(--background)',
                    }
                    : {
                        height: 'calc(100vh - 120px)',
                        minHeight: 780,
                        background: 'var(--background)',
                        borderRadius: 0,
                    }
            }
        >
            <svg ref={svgRef} className="w-full h-full" />

            {/* Tooltip overlay */}
            <div
                ref={tooltipRef}
                style={{
                    position: 'fixed',
                    zIndex: 50,
                    pointerEvents: 'none',
                    opacity: 0,
                    transition: 'opacity 0.08s ease',
                    maxWidth: 360,
                    padding: 10,
                    borderRadius: 10,
                    background: 'rgba(26,26,26,0.96)',
                    border: '1px solid rgba(51,51,51,0.9)',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.35)',
                    fontFamily: 'Inter, system-ui, sans-serif',
                }}
            />
        </div>
    );

    if (fullscreen) return graphEl;

    return (
        <section className="w-full mt-8">
            <div className="w-full">
                {/* Full-bleed graph area (fills remaining viewport under header/search). */}
                {graphEl}

                {/* Title/subtitle moved below the graph */}
                <div className="mt-6 px-1 pb-2">
                    <h2 className="text-lg sm:text-xl font-semibold" style={{ color: 'var(--primary-text)' }}>
                        Citation neighborhoods (paper {ARROW} paper)
                    </h2>
                    <p className="mt-1 text-sm" style={{ color: 'var(--secondary-text)' }}>
                        Each island is a connected component of the arXiv citation graph inferred from bibliography matches.
                        Click a paper to open its artifact dependency graph.
                    </p>

                    {loadError ? (
                        <div className="mt-3 text-sm" style={{ color: '#ff6b6b' }}>
                            Failed to load citation components: {loadError}
                        </div>
                    ) : null}

                    {loading ? (
                        <div className="mt-3 text-sm" style={{ color: 'var(--secondary-text)' }}>
                            Loading citation components…
                        </div>
                    ) : null}
                </div>
            </div>
        </section>
    );
}
