import * as d3 from 'd3';

import { COLORS } from './config';
import { processGraphData } from './data';
import { setupDrag, setupInteractions } from './interaction';
import { setupLegends } from './ui';
import type { ConstellationEdge, ConstellationNode } from './types';

export type IncrementalGraphState = {
    svg: any;
    defs: any;
    g: any;

    linkLayer: any;
    nodeLayer: any;
    labelLayer: any;

    simulation: d3.Simulation<any, any>;
    linkForce: d3.ForceLink<any, any>;

    nodeById: Map<string, any>;
    edgeByKey: Map<string, any>;

    linkSel: any;
    nodeSel: any;
    labelSel: any;

    nodeDegrees: Map<string, number>;
    radiusScale: any;

    dragBehavior: d3.DragBehavior<SVGCircleElement, any, any>;

    rawGraphData: { nodes: any[]; edges: any[] };

    // cached colors/types
    nodeColors: Record<string, string>;
    edgeColors: Record<string, string>;
};

export type InitIncrementalGraphArgs = {
    svgEl: SVGSVGElement;
    width: number;
    height: number;
    state: any;
    actions: any;
};

function edgeKey(e: ConstellationEdge) {
    const s = typeof e.source === 'object' ? e.source.id : e.source;
    const t = typeof e.target === 'object' ? e.target.id : e.target;

    const dep = (e as any).dependency_type || (e as any).dependencyType || 'internal';
    const ref = (e as any).reference_type || (e as any).referenceType || '';
    const typ = (e as any).type || '';

    return `${s}=>${t}::${dep}::${ref}::${typ}`;
}

function computeDegrees(edges: any[]) {
    const nodeDegrees = new Map<string, number>();
    edges.forEach((e: any) => {
        const s = typeof e.source === 'object' ? e.source.id : e.source;
        const t = typeof e.target === 'object' ? e.target.id : e.target;
        nodeDegrees.set(s, (nodeDegrees.get(s) || 0) + 1);
        nodeDegrees.set(t, (nodeDegrees.get(t) || 0) + 1);
    });
    return nodeDegrees;
}

function makeRadiusScale(nodeDegrees: Map<string, number>) {
    const degreeValues = Array.from(nodeDegrees.values());
    // Ensure the scale has a reasonable spread even when most nodes have low degree
    const maxDeg = degreeValues.length ? Math.max(3, ...degreeValues) : 3;
    const minRadius = 10;
    const maxRadius = 24;
    return d3.scaleSqrt().domain([1, maxDeg]).range([minRadius, maxRadius]);
}

function ensureEdgeMarkers(defs: d3.Selection<SVGDefsElement, unknown, null, undefined>, edgeTypes: string[], edgeColors: Record<string, string>) {
    edgeTypes.forEach((type) => {
        const id = `arrowhead-${type}`;
        if (!defs.select(`#${id}`).node()) {
            defs.append('marker')
                .attr('id', id)
                .attr('viewBox', '-0 -5 10 10')
                .attr('refX', 10)
                .attr('refY', 0)
                .attr('orient', 'auto')
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', edgeColors[type]);
        }
    });
}

export function initIncrementalGraph({ svgEl, width, height, state, actions }: InitIncrementalGraphArgs): IncrementalGraphState {
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const defs = svg.append('defs');
    const g = svg.append('g');

    const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom as any);

    const linkLayer = g.append('g');
    const nodeLayer = g.append('g');
    const labelLayer = g.append('g');

    const linkForce = d3
        .forceLink<any, any>([] as any)
        .id((d: any) => d.id)
        .distance(90)
        .strength(0.6);

    const simulation = d3
        .forceSimulation<any>([] as any)
        .force('link', linkForce)
        .force('charge', d3.forceManyBody().strength(-250))
        .force('center', d3.forceCenter(width / 2, height / 2));

    const nodeById = new Map<string, any>();
    const edgeByKey = new Map<string, any>();

    const nodeDegrees = new Map<string, number>();
    const radiusScale = makeRadiusScale(nodeDegrees);

    simulation.force('collide', d3.forceCollide().radius((d: any) => radiusScale(nodeDegrees.get(d.id) || 1) + 6));

    const linkSel = linkLayer.selectAll<SVGLineElement, any>('line');
    const nodeSel = nodeLayer.selectAll<SVGCircleElement, any>('circle');
    const labelSel = labelLayer.selectAll<SVGTextElement, any>('text');

    const dragBehavior = setupDrag(simulation);

    const rawGraphData = { nodes: [] as any[], edges: [] as any[] };

    // minimal initial refs
    state.refs = {
        svg,
        g,
        simulation,
        link: linkSel,
        node: nodeSel,
        label: labelSel,
        nodeById: new Map(),
        outgoingEdgesBySource: new Map(),
        incomingEdgesByTarget: new Map(),
        graphData: rawGraphData,
    };

    setupInteractions(nodeSel as any, linkSel as any, labelSel as any, svg as any, state, actions);

    return {
        svg,
        defs,
        g,
        linkLayer,
        nodeLayer,
        labelLayer,
        simulation,
        linkForce,
        nodeById,
        edgeByKey,
        linkSel,
        nodeSel,
        labelSel,
        nodeDegrees,
        radiusScale,
        dragBehavior,
        rawGraphData,
        nodeColors: {},
        edgeColors: {},
    };
}

export function upsertNode(ig: IncrementalGraphState, node: ConstellationNode): { changed: boolean } {
    const existing = ig.nodeById.get(node.id);
    if (existing) {
        // mutate in place to preserve positions (x/y/fx/fy)
        Object.assign(existing, node);
        return { changed: true };
    }

    ig.nodeById.set(node.id, { ...node });
    return { changed: true };
}

export function addEdge(ig: IncrementalGraphState, edge: ConstellationEdge): { added: boolean } {
    const k = edgeKey(edge);
    if (ig.edgeByKey.has(k)) return { added: false };
    ig.edgeByKey.set(k, { ...edge });
    return { added: true };
}

export function applyMutations(ig: IncrementalGraphState, state: any, actions: any) {
    const nodes = Array.from(ig.nodeById.values());
    const edges = Array.from(ig.edgeByKey.values());

    const processed = processGraphData({ nodes, edges });

    ig.rawGraphData.nodes = processed.nodes;
    ig.rawGraphData.edges = processed.edges;

    ig.nodeColors = processed.nodeColors;
    ig.edgeColors = processed.edgeColors;

    ensureEdgeMarkers(ig.defs, processed.edgeTypes, processed.edgeColors);

    ig.nodeDegrees = computeDegrees(processed.edges);
    ig.radiusScale = makeRadiusScale(ig.nodeDegrees);

    ig.simulation.force(
        'collide',
        d3.forceCollide().radius((d: any) => ig.radiusScale(ig.nodeDegrees.get(d.id) || 1) + 6),
    );

    ig.simulation.nodes(processed.nodes as any);
    ig.linkForce.links(processed.edges as any);

    // links
    ig.linkSel = ig.linkLayer
        .selectAll('line')
        .data(processed.edges as any, (d: any) => {
            const s = typeof d.source === 'object' ? d.source.id : d.source;
            const t = typeof d.target === 'object' ? d.target.id : d.target;
            return `${s}=>${t}::${d.dependency_type || d.dependencyType || 'internal'}`;
        })
        .join(
            (enter: any) =>
                enter
                    .append('line')
                    .attr('class', 'link')
                    .attr('stroke', (d: any) => processed.edgeColors[d.dependency_type || 'internal'] || COLORS.edges('internal'))
                    .attr('marker-end', (d: any) => `url(#arrowhead-${d.dependency_type || 'internal'})`),
            (update: any) =>
                update
                    .attr('stroke', (d: any) => processed.edgeColors[d.dependency_type || 'internal'] || COLORS.edges('internal'))
                    .attr('marker-end', (d: any) => `url(#arrowhead-${d.dependency_type || 'internal'})`),
            (exit: any) => exit.remove(),
        );

    // nodes
    const entered = new Set<string>();
    ig.nodeSel = ig.nodeLayer
        .selectAll('circle')
        .data(processed.nodes as any, (d: any) => d.id)
        .join(
            (enter: any) =>
                enter
                    .append('circle')
                    .attr('class', 'node')
                    .each((d: any) => entered.add(d.id)),
            (update: any) => update,
            (exit: any) => exit.remove(),
        )
        .attr('r', (d: any) => ig.radiusScale(ig.nodeDegrees.get(d.id) || 1))
        .attr('fill', (d: any) => processed.nodeColors[d.type] || '#ccc');

    // labels
    ig.labelSel = ig.labelLayer
        .selectAll('text')
        .data(processed.nodes as any, (d: any) => d.id)
        .join(
            (enter: any) => enter.append('text').attr('class', 'node-label'),
            (update: any) => update,
            (exit: any) => exit.remove(),
        )
        // Keep labels close to nodes, with a small padding
        .attr('dy', (d: any) => ig.radiusScale(ig.nodeDegrees.get(d.id) || 1) + 6)
        .text((d: any) => d.display_name || d.label || d.id);

    if (entered.size) {
        ig.nodeSel.filter((d: any) => entered.has(d.id)).call(ig.dragBehavior as any);
    }

    // Update refs for existing interaction/ui modules
    state.graphData = ig.rawGraphData;
    state.refs.nodeById = processed.nodeById;
    state.refs.outgoingEdgesBySource = processed.outgoingEdgesBySource;
    state.refs.incomingEdgesByTarget = processed.incomingEdgesByTarget;
    state.refs.link = ig.linkSel;
    state.refs.node = ig.nodeSel;
    state.refs.label = ig.labelSel;

    setupInteractions(ig.nodeSel as any, ig.linkSel as any, ig.labelSel as any, ig.svg as any, state, actions);
    setupLegends(processed.nodeTypes, processed.edgeTypes, processed.nodeColors, processed.edgeColors, state, actions);

    ig.simulation.on('tick', () => {
        // Draw links shortened to the node boundaries so arrowheads remain visible.
        ig.linkSel
            .attr('x1', (d: any) => {
                const r = ig.radiusScale(ig.nodeDegrees.get(d.source.id) || 1);
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                return d.source.x + (dx / dist) * r;
            })
            .attr('y1', (d: any) => {
                const r = ig.radiusScale(ig.nodeDegrees.get(d.source.id) || 1);
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                return d.source.y + (dy / dist) * r;
            })
            .attr('x2', (d: any) => {
                const r = ig.radiusScale(ig.nodeDegrees.get(d.target.id) || 1);
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                return d.target.x - (dx / dist) * r;
            })
            .attr('y2', (d: any) => {
                const r = ig.radiusScale(ig.nodeDegrees.get(d.target.id) || 1);
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                return d.target.y - (dy / dist) * r;
            });

        ig.nodeSel.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y);

        ig.labelSel
            .attr('x', (d: any) => d.x)
            .attr('y', (d: any) => d.y + ig.radiusScale(ig.nodeDegrees.get(d.id) || 1) + 6);
    });

    ig.simulation.alpha(0.25).restart();
}
