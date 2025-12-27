import { edgeKey } from './data';

export function getMaxPrereqDepth(
    startId: string,
    outgoingEdgesBySource: Map<string, Array<{ s: string; t: string; dep: string }>>,
) {
    const visited = new Set([startId]);
    let frontier = [startId];
    let depth = 0;

    while (frontier.length) {
        const next: string[] = [];
        for (const id of frontier) {
            const outs = outgoingEdgesBySource.get(id) || [];
            for (const { t } of outs) {
                if (!visited.has(t)) {
                    visited.add(t);
                    next.push(t);
                }
            }
        }
        if (next.length === 0) break;
        depth += 1;
        frontier = next;
    }

    return depth;
}

export function recomputeProofSubgraph(
    state: any,
    incomingEdgesByTarget: Map<string, Array<{ s: string; t: string; dep: string }>>,
) {
    state.proofVisibleNodes = new Set([state.proofTargetId]);
    state.proofVisibleEdges = new Set<string>();

    let frontier = [state.proofTargetId];
    let level = 0;
    while (level < state.proofDepth && frontier.length) {
        const next: string[] = [];
        for (const id of frontier) {
            const ins = incomingEdgesByTarget.get(id) || [];
            for (const { s, t, dep } of ins) {
                if (dep === 'generalized_by') continue;
                state.proofVisibleNodes.add(s);
                state.proofVisibleEdges.add(edgeKey(s, t));
                next.push(s);
            }
        }
        level += 1;
        frontier = next;
    }
}
