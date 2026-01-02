import { COLORS, NODE_PALETTE } from './config';
import type { ConstellationEdge, ConstellationGraphData, ConstellationNode } from './types';

export type ProcessedGraph = {
    nodes: ConstellationNode[];
    edges: ConstellationEdge[];
    nodeTypes: string[];
    edgeTypes: string[];
    nodeColors: Record<string, string>;
    edgeColors: Record<string, string>;
    nodeById: Map<string, ConstellationNode>;
    outgoingEdgesBySource: Map<string, Array<{ s: string; t: string; dep: string }>>;
    incomingEdgesByTarget: Map<string, Array<{ s: string; t: string; dep: string }>>;
};

// Same semantic normalization as constellations/assets/modules/data.js
export function processGraphData(graphData: ConstellationGraphData): ProcessedGraph {
    const edges = (graphData.edges || [])
        .map((e) => {
            const dep = e.dependency_type || 'internal';
            const ref = (e as any).reference_type || (e as any).referenceType || null;
            const typ = (e as any).type || null;

            // Canonical semantics:
            // We render edges as SVG arrows from `source -> target`.
            // For `used_in`, we want the arrow to mean:
            //   prerequisite -> dependent
            // So `source` should already be the prerequisite and `target` the dependent.
            // (We normalize backend semantics in the API layer, not here.)
            // - For legacy/raw forms like "uses_result" / "uses_definition" /
            //   "is_corollary_of", we flip them into the same prerequisite ->
            //   result orientation and re-label as "used_in".
            if (dep === 'used_in') {
                return { ...e, dependency_type: 'used_in' };
            }
            if (dep === 'uses_result' || dep === 'uses_definition' || dep === 'is_corollary_of') {
                return { ...e, dependency_type: 'used_in', source: (e as any).target, target: (e as any).source };
            }
            if (dep === 'is_generalization_of' || dep === 'generalized_by') {
                return { ...e, dependency_type: 'generalized_by', source: (e as any).target, target: (e as any).source };
            }

            // Internal cross-references (\Cref, \ref, etc.) are also dependencies:
            // a node that references another node depends on it, so we want
            // prerequisite -> result (referenced -> referrer).
            if (dep === 'internal' && (ref === 'internal' || typ === 'internal')) {
                return { ...e, dependency_type: 'internal', source: (e as any).target, target: (e as any).source };
            }
            if (dep === 'provides_remark') {
                return null;
            }
            return e;
        })
        .filter(Boolean) as ConstellationEdge[];

    const nodes = graphData.nodes || [];

    const nodeTypes = Array.from(new Set(nodes.map((d) => d.type)));
    const edgeTypes = Array.from(new Set(edges.map((d) => d.dependency_type || 'internal')));

    // Match Constellations: stable semantic order so colors are consistent across papers.
    const CANONICAL_NODE_TYPE_ORDER = [
        'theorem',
        'lemma',
        'proposition',
        'corollary',
        'definition',
        'remark',
        'conjecture',
        'assumption',
        'proof',
        'example',
        'claim',
        'fact',
        'observation',
        'unknown',
    ];

    const orderedNodeTypes = [
        ...CANONICAL_NODE_TYPE_ORDER.filter((t) => nodeTypes.includes(t)),
        ...nodeTypes.filter((t) => !CANONICAL_NODE_TYPE_ORDER.includes(t)).sort(),
    ];

    const nodeColors = orderedNodeTypes.reduce<Record<string, string>>((acc, type, i) => {
        acc[type] = NODE_PALETTE[i % NODE_PALETTE.length];
        return acc;
    }, {});

    const edgeColors = edgeTypes.reduce<Record<string, string>>((acc, type) => {
        acc[type] = COLORS.edges(type);
        return acc;
    }, {});

    const nodeById = new Map(nodes.map((n) => [n.id, n]));
    const outgoingEdgesBySource = new Map<string, Array<{ s: string; t: string; dep: string }>>();
    const incomingEdgesByTarget = new Map<string, Array<{ s: string; t: string; dep: string }>>();

    edges.forEach((e) => {
        const s = typeof e.source === 'object' ? e.source.id : e.source;
        const t = typeof e.target === 'object' ? e.target.id : e.target;
        const dep = e.dependency_type || 'internal';

        if (!outgoingEdgesBySource.has(s)) outgoingEdgesBySource.set(s, []);
        outgoingEdgesBySource.get(s)!.push({ s, t, dep });

        if (!incomingEdgesByTarget.has(t)) incomingEdgesByTarget.set(t, []);
        incomingEdgesByTarget.get(t)!.push({ s, t, dep });
    });

    return {
        nodes,
        edges,
        nodeTypes,
        edgeTypes,
        nodeColors,
        edgeColors,
        nodeById,
        outgoingEdgesBySource,
        incomingEdgesByTarget,
    };
}

export function edgeKey(s: string, t: string) {
    return `${s}=>${t}`;
}
