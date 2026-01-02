import { edgeKey } from './data';

/**
 * Compute the maximum prerequisite depth reachable *upstream* from a node.
 *
 * This must follow the same direction as `recomputeProofSubgraph`, i.e.
 * walking backwards along incoming edges (prerequisite -> result), so that
 * `proofDepth` and the visual unfolding depth stay in sync.
 */
export function getMaxPrereqDepth(
    startId: string,
    incomingEdgesByTarget: Map<string, Array<{ s: string; t: string; dep: string }>>,
) {
    const visited = new Set([startId]);
    let frontier = [startId];
    let depth = 0;

    while (frontier.length) {
        const next: string[] = [];
        for (const id of frontier) {
            const ins = incomingEdgesByTarget.get(id) || [];
            for (const { s, dep } of ins) {
                // Keep this aligned with recomputeProofSubgraph's traversal
                if (dep === 'generalized_by') continue;
                if (!visited.has(s)) {
                    visited.add(s);
                    next.push(s);
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
