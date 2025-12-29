'use client';

import * as d3 from 'd3';
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';

import { getMaxPrereqDepth, recomputeProofSubgraph } from './constellations/proof';
import { edgeKey } from './constellations/data';

import {
    hideInfoPanel,
    hideTooltip,
    renderNodeTooltip,
    updateInfoPanel,
} from './constellations/ui';

import {
    addEdge,
    applyMutations,
    initIncrementalGraph,
    upsertNode,
    type IncrementalGraphState,
} from './constellations/incremental';

import type { ConstellationEdge, ConstellationNode } from './constellations/types';

type Props = {
    nodes?: ConstellationNode[];
    links?: ConstellationEdge[];
    onReportNode?: (node: { id: string; label: string; type?: string }) => void;
};

export type GraphIngestEvent = { type: string; data?: any };

export type ConstellationsGraphHandle = {
    ingest: (ev: GraphIngestEvent) => void;
    reset: () => void;
};

const ConstellationsGraph = forwardRef<ConstellationsGraphHandle, Props>(function ConstellationsGraph(
    { nodes = [], links = [], onReportNode },
    ref,
) {
    const containerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);

    const tooltipRef = useRef<HTMLDivElement>(null);
    const infoPanelRef = useRef<HTMLDivElement>(null);
    const infoTitleRef = useRef<HTMLDivElement>(null);
    const infoBodyRef = useRef<HTMLDivElement>(null);

    const [legendOpen, setLegendOpen] = useState(true);

    const [state] = useState(() => ({
        pinned: false,
        pinnedNode: null as any,
        hiddenTypes: new Set<string>(),
        proofMode: false,
        proofTargetId: null as string | null,
        proofDepth: 1,
        proofVisibleNodes: new Set<string>(),
        proofVisibleEdges: new Set<string>(),
        graphData: null as any,
        refs: null as any,
    }));

    const igRef = useRef<IncrementalGraphState | null>(null);

    // Initial load (non-streaming): ingest given arrays once.
    useEffect(() => {
        if (!nodes.length && !links.length) return;
        if (!igRef.current) return;

        for (const n of nodes) upsertNode(igRef.current, n);
        for (const e of links) addEdge(igRef.current, e);

        applyMutations(igRef.current, state, actions);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Actions used by interaction/ui
    const actions = useMemo(() => {
        return {
            // Kept for compatibility with the original Constellations interaction module.
            // We moved the proof controls into the right-side info panel.
            updateFloatingControls: () => { },

            updateVisibility: () => {
                const { node, label, link, simulation, graphData } = state.refs;
                node.style('display', (d: any) => (state.hiddenTypes.has(d.type) ? 'none' : null));
                label.style('display', (d: any) => (state.hiddenTypes.has(d.type) ? 'none' : null));
                link.style('display', (d: any) => {
                    const sType = typeof d.source === 'object'
                        ? d.source.type
                        : graphData.nodes.find((n: any) => n.id === d.source)?.type;
                    const tType = typeof d.target === 'object'
                        ? d.target.type
                        : graphData.nodes.find((n: any) => n.id === d.target)?.type;

                    const sourceVisible = !state.hiddenTypes.has(sType);
                    const targetVisible = !state.hiddenTypes.has(tType);
                    return sourceVisible && targetVisible ? null : 'none';
                });
                if (!state.pinned) simulation.alpha(0.3).restart();
            },

            renderNodeTooltip: (event: any, d: any) => {
                if (tooltipRef.current) renderNodeTooltip(tooltipRef.current, event, d);
            },
            hideTooltip: () => {
                if (tooltipRef.current) hideTooltip(tooltipRef.current);
            },

            updateInfoPanel: (d: any) => {
                if (infoPanelRef.current && infoTitleRef.current && infoBodyRef.current) {
                    updateInfoPanel(infoPanelRef.current, infoTitleRef.current, infoBodyRef.current, d, state, actions);
                }
            },
            hideInfoPanel: () => {
                if (infoPanelRef.current) hideInfoPanel(infoPanelRef.current);
            },

            enterProofMode: (targetId: string) => {
                state.proofMode = true;
                state.proofTargetId = targetId;
                state.proofDepth = 1;
                state.pinned = true;
                state.pinnedNode = state.refs.nodeById.get(targetId) || null;

                actions.hideTooltip();
                state.refs.node.classed('selected', (n: any) => n.id === targetId);

                actions.recomputeProofSubgraph();
                if (state.refs.nodeById.has(targetId)) actions.updateInfoPanel(state.refs.nodeById.get(targetId));
            },

            exitProofMode: () => {
                state.proofMode = false;
                state.proofTargetId = null;
                state.proofVisibleNodes = new Set();
                state.proofVisibleEdges = new Set();

                state.pinned = false;
                state.pinnedNode = null;
                state.refs.node.classed('selected', false);
                actions.hideInfoPanel();
                actions.updateVisibility();
            },

            recomputeProofSubgraph: () => {
                recomputeProofSubgraph(state, state.refs.incomingEdgesByTarget);
                const { node, label, link, simulation } = state.refs;
                node.style('display', (d: any) => (state.proofVisibleNodes.has(d.id) ? null : 'none'));
                label.style('display', (d: any) => (state.proofVisibleNodes.has(d.id) ? null : 'none'));
                link.style('display', (l: any) => {
                    const sId = typeof l.source === 'object' ? l.source.id : l.source;
                    const tId = typeof l.target === 'object' ? l.target.id : l.target;
                    return state.proofVisibleEdges.has(edgeKey(sId, tId)) ? null : 'none';
                });
                simulation.alpha(0.3).restart();
            },

            unfoldLess: () => {
                if (!state.proofMode) return;
                state.proofDepth = Math.max(1, state.proofDepth - 1);
                actions.recomputeProofSubgraph();
                if (state.proofTargetId) actions.updateInfoPanel(state.refs.nodeById.get(state.proofTargetId));
            },

            unfoldMore: () => {
                if (!state.proofMode) return;
                state.proofDepth = Math.min(
                    getMaxPrereqDepth(state.proofTargetId!, state.refs.outgoingEdgesBySource),
                    state.proofDepth + 1,
                );
                actions.recomputeProofSubgraph();
                if (state.proofTargetId) actions.updateInfoPanel(state.refs.nodeById.get(state.proofTargetId));
            },

            reportNodeIssue: (d: any) => {
                if (!onReportNode) return;
                const label = String(d?.display_name ?? d?.label ?? d?.id ?? '');
                const id = String(d?.id ?? '');
                const type = d?.type ? String(d.type) : undefined;
                if (!id) return;
                onReportNode({ id, label, type });
            },
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [state, onReportNode]);

    useEffect(() => {
        if (!svgRef.current) return;

        const rect = svgRef.current.getBoundingClientRect();
        igRef.current = initIncrementalGraph({
            svgEl: svgRef.current,
            width: rect.width,
            height: rect.height,
            state,
            actions,
        });

        // Close panel button
        const closeBtn = document.getElementById('close-info-panel');
        if (closeBtn)
            closeBtn.onclick = () => {
                if (state.proofMode) {
                    actions.exitProofMode();
                    return;
                }
                if (state.pinned) {
                    state.pinned = false;
                    state.pinnedNode = null;
                    state.refs.node.classed('selected', false);
                    actions.hideInfoPanel();
                    actions.updateVisibility();
                }
            };

        return () => {
            igRef.current?.simulation.stop();
            d3.select(svgRef.current).selectAll('*').remove();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useImperativeHandle(ref, () => ({
        ingest: (ev: GraphIngestEvent) => {
            if (!igRef.current) return;
            if (!ev || !ev.type) return;

            if (ev.type === 'node' && ev.data) {
                upsertNode(igRef.current, ev.data as any);
                applyMutations(igRef.current, state, actions);

                // If a node is pinned, refresh info panel so prerequisites update live.
                if (state.pinned && state.pinnedNode && state.pinnedNode.id === ev.data.id) {
                    actions.updateInfoPanel(state.pinnedNode);
                }
                return;
            }

            if (ev.type === 'link' && ev.data) {
                const { added } = addEdge(igRef.current, ev.data as any);
                if (added) {
                    applyMutations(igRef.current, state, actions);
                }
                return;
            }

            if (ev.type === 'reset') {
                igRef.current.nodeById.clear();
                igRef.current.edgeByKey.clear();
                applyMutations(igRef.current, state, actions);
            }
        },
        reset: () => {
            if (!igRef.current) return;
            igRef.current.nodeById.clear();
            igRef.current.edgeByKey.clear();
            applyMutations(igRef.current, state, actions);
        },
    }));

    return (
        <div className="graph-shell" ref={containerRef}>
            <div className="graph-container">
                <svg ref={svgRef} id="graph" className="w-full h-full" />

                <div id="tooltip" ref={tooltipRef} />

                <div id="info-panel" ref={infoPanelRef}>
                    <div id="info-header" className="info-header">
                        <div id="info-title" ref={infoTitleRef} />
                        <button id="close-info-panel" className="close-btn" aria-label="Close">×</button>
                    </div>
                    <div id="info-body" className="info-content" ref={infoBodyRef} />
                </div>

                <div className={legendOpen ? 'legend' : 'legend legend--collapsed'}>
                    <div className="legend-header">
                        <h3 className="legend-title">Legend</h3>
                        <button
                            type="button"
                            className="legend-toggle"
                            aria-label={legendOpen ? 'Collapse legend' : 'Expand legend'}
                            onClick={() => setLegendOpen((v) => !v)}
                        >
                            {legendOpen ? '▾' : '▸'}
                        </button>
                    </div>

                    <div className="legend-body">
                        <h3>Node Types</h3>
                        <div id="node-legend-container" className="legend-grid" />
                        <h3 style={{ marginTop: 10 }}>Edge Types</h3>
                        <div id="edge-legend-container" className="legend-grid" />
                        <p style={{ marginTop: 8, fontSize: 12, color: 'var(--secondary-text)' }}>
                            Click a node to focus. Right-click a node to explore proof path.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
});

export default ConstellationsGraph;
