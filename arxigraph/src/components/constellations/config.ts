export const ZOOM_EXTENT: [number, number] = [0.1, 8];

// d3.schemeCategory10
export const NODE_PALETTE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
];

export const COLORS = {
    // Node colors are assigned in `processGraphData` using NODE_PALETTE order.
    nodes: () => NODE_PALETTE[0],
    edges: (type: string) => {
        // closer to constellations defaults
        switch (type) {
            case 'used_in':
                return '#1f77b4';
            case 'generalized_by':
                return '#9467bd';
            case 'internal':
            default:
                return '#999';
        }
    },
};
