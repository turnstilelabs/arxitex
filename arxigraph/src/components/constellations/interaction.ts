import * as d3 from 'd3';

export function setupDrag(simulation: d3.Simulation<any, any>) {
    function dragstarted(event: any, d: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event: any, d: any) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event: any, d: any) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    return d3.drag<SVGCircleElement, any>().on('start', dragstarted).on('drag', dragged).on('end', dragended);
}

export function setupInteractions(
    node: d3.Selection<SVGCircleElement, any, SVGGElement, unknown>,
    link: d3.Selection<SVGLineElement, any, SVGGElement, unknown>,
    label: d3.Selection<SVGTextElement, any, SVGGElement, unknown>,
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    state: any,
    actions: any,
) {
    node.on('contextmenu', (event: any, d: any) => {
        event.preventDefault();
        actions.enterProofMode(d.id);
    });

    node.on('click', (event: any, d: any) => {
        event.stopPropagation();

        if (state.proofMode) {
            state.pinned = true;
            state.pinnedNode = d;
            actions.hideTooltip();
            node.classed('selected', (n: any) => n.id === d.id);
            actions.updateInfoPanel(d);
            actions.updateFloatingControls();
            return;
        }

        state.pinned = true;
        state.pinnedNode = d;
        actions.hideTooltip();
        node.classed('selected', (n: any) => n.id === d.id);

        const subgraphNodes = new Set<string>([d.id]);
        state.graphData.edges.forEach((edge: any) => {
            const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
            const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
            if (sourceId === d.id) subgraphNodes.add(targetId);
            if (targetId === d.id) subgraphNodes.add(sourceId);
        });

        node.style('display', (n: any) => (subgraphNodes.has(n.id) ? null : 'none'));
        label.style('display', (n: any) => (subgraphNodes.has(n.id) ? null : 'none'));

        link.style('display', (l: any) => {
            const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
            const targetId = typeof l.target === 'object' ? l.target.id : l.target;
            return subgraphNodes.has(sourceId) && subgraphNodes.has(targetId) ? null : 'none';
        });

        actions.updateInfoPanel(d);
        actions.updateFloatingControls();
    });

    svg.on('click', () => {
        if (state.proofMode) {
            actions.exitProofMode();
            return;
        }

        if (state.pinned) {
            state.pinned = false;
            state.pinnedNode = null;
            node.classed('selected', false);
            actions.hideInfoPanel();
            actions.updateVisibility();
            actions.updateFloatingControls();
        }
    });

    node.on('mouseover', (event: any, d: any) => actions.renderNodeTooltip(event, d));
    node.on('mouseout', () => {
        if (!state.pinned) actions.hideTooltip();
    });
    link.on('mouseout', () => {
        if (!state.pinned) actions.hideTooltip();
    });
}
