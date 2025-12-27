export type ConstellationNode = {
    id: string;
    type: string;
    display_name?: string;
    content?: string;
    content_preview?: string;
    prerequisites_preview?: string;
    label?: string | null;
    position?: {
        line_start?: number;
        line_end?: number | null;
        col_start?: number | null;
        col_end?: number | null;
    };
    references?: any[];
    proof?: string | null;
    // d3 simulation fields
    x?: number;
    y?: number;
    fx?: number | null;
    fy?: number | null;
};

export type ConstellationEdge = {
    source: string | ConstellationNode;
    target: string | ConstellationNode;
    dependency_type?: string | null;
    reference_type?: string | null;
    type?: string | null;
    context?: string | null;
    dependency?: string | null;
};

export type ConstellationGraphData = {
    nodes: ConstellationNode[];
    edges: ConstellationEdge[];
};
