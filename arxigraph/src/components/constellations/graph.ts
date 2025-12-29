import * as d3 from 'd3';
import { ZOOM_EXTENT } from './config';

export function initializeGraph(
    svgEl: SVGSVGElement,
    edgeTypes: string[],
    edgeColors: Record<string, string>,
) {
    const svg = d3.select(svgEl);
    const width = svgEl.getBoundingClientRect().width;
    const height = svgEl.getBoundingClientRect().height;

    svg.selectAll('*').remove();

    const defs = svg.append('defs');
    edgeTypes.forEach((type) => {
        defs.append('marker')
            .attr('id', `arrowhead-${type}`)
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 10)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', edgeColors[type]);
    });

    const g = svg.append('g');

    const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent(ZOOM_EXTENT)
        .on('zoom', (event) => g.attr('transform', event.transform));

    svg.call(zoom as any);

    return { svg, g, width, height };
}

export function renderElements(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    nodes: any[],
    edges: any[],
    nodeColors: Record<string, string>,
    edgeColors: Record<string, string>,
    radiusScale: any,
    nodeDegrees: Map<string, number>,
) {
    const link = g
        .append('g')
        .selectAll('line')
        .data(edges)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke', (d: any) => edgeColors[d.dependency_type || 'internal'])
        .attr('marker-end', (d: any) => `url(#arrowhead-${d.dependency_type || 'internal'})`);

    const node = g
        .append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('r', (d: any) => radiusScale(nodeDegrees.get(d.id) || 1))
        .attr('fill', (d: any) => nodeColors[d.type] || '#ccc');

    const label = g
        .append('g')
        .selectAll('text')
        .data(nodes)
        .enter()
        .append('text')
        .attr('class', 'node-label')
        // Position labels just below the node, with a small padding
        .attr('dy', (d: any) => radiusScale(nodeDegrees.get(d.id) || 1) + 4)
        .text((d: any) => d.display_name || d.label || d.id);

    return { link, node, label };
}
