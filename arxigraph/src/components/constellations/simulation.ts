import * as d3 from 'd3';
import type { ConstellationEdge, ConstellationNode } from './types';

export function setupSimulation(
    nodes: ConstellationNode[],
    edges: ConstellationEdge[],
    width: number,
    height: number,
) {
    const nodeDegrees = new Map<string, number>();
    edges.forEach((e) => {
        const s = typeof e.source === 'object' ? e.source.id : e.source;
        const t = typeof e.target === 'object' ? e.target.id : e.target;
        nodeDegrees.set(s, (nodeDegrees.get(s) || 0) + 1);
        nodeDegrees.set(t, (nodeDegrees.get(t) || 0) + 1);
    });

    const degreeValues = Array.from(nodeDegrees.values());
    const maxDeg = degreeValues.length ? Math.max(3, ...degreeValues) : 3;
    const minRadius = 10;
    const maxRadius = 24;
    const radiusScale = d3.scaleSqrt().domain([1, maxDeg]).range([minRadius, maxRadius]);

    const simulation = d3
        .forceSimulation(nodes as any)
        .force(
            'link',
            d3
                .forceLink(edges as any)
                .id((d: any) => d.id)
                .distance(90)
                .strength(0.6),
        )
        .force('charge', d3.forceManyBody().strength(-250))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide().radius((d: any) => radiusScale(nodeDegrees.get(d.id) || 1) + 6));

    return { simulation, radiusScale, nodeDegrees };
}

export function updateSimulationTick(
    simulation: d3.Simulation<any, any>,
    link: d3.Selection<SVGLineElement, any, SVGGElement, unknown>,
    node: d3.Selection<SVGCircleElement, any, SVGGElement, unknown>,
    label: d3.Selection<SVGTextElement, any, SVGGElement, unknown>,
    radiusScale: any,
    nodeDegrees: Map<string, number>,
) {
    simulation.on('tick', () => {
        link
            .attr('x1', (d: any) => d.source.x)
            .attr('y1', (d: any) => d.source.y)
            .attr('x2', (d: any) => d.target.x)
            .attr('y2', (d: any) => d.target.y);

        node.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y);

        label
            .attr('x', (d: any) => d.x)
            .attr('y', (d: any) => d.y + radiusScale(nodeDegrees.get(d.id) || 1) + 6);
    });
}
