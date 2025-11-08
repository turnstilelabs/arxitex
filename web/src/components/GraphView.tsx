"use client";

import CytoscapeComponent from "react-cytoscapejs";
import Cytoscape, { ElementDefinition } from "cytoscape";
import dagre from "cytoscape-dagre";
import type { ArtifactNode, DependencyType, DocumentGraph, Edge } from "@/lib/types";
import { useEffect, useMemo, useRef, useState } from "react";

// Register layout
// @ts-ignore - plugin typing
Cytoscape.use(dagre);

type Props = {
    graph: DocumentGraph;
    onSelectNode?: (node: ArtifactNode | null) => void;
    height?: string | number;
};

// Basic color mapping per artifact type
const NODE_COLORS: Record<string, string> = {
    theorem: "#2563eb",
    lemma: "#0891b2",
    definition: "#059669",
    proposition: "#7c3aed",
    corollary: "#db2777",
    example: "#ea580c",
    remark: "#6b7280",
    conjecture: "#a16207",
    claim: "#0d9488",
    fact: "#9333ea",
    observation: "#3f6212",
    unknown: "#94a3b8",
};

// Edge style by dependency/reference type
function edgeColor(type: string | null | undefined): string {
    if (!type) return "#9ca3af";
    switch (type) {
        case "uses_result":
            return "#0ea5e9";
        case "uses_definition":
            return "#22c55e";
        case "proves":
            return "#f59e0b";
        case "provides_example":
            return "#ef4444";
        case "provides_remark":
            return "#94a3b8";
        case "is_corollary_of":
            return "#e879f9";
        case "is_special_case_of":
            return "#a3e635";
        case "is_generalization_of":
            return "#f472b6";
        case "internal":
            return "#9ca3af";
        default:
            return "#9ca3af";
    }
}

function toElements(nodes: ArtifactNode[], edges: Edge[]): ElementDefinition[] {
    const els: ElementDefinition[] = [];

    for (const n of nodes) {
        els.push({
            data: {
                id: n.id,
                label: n.display_name,
                type: n.type,
                raw: n,
            },
        });
    }

    for (const e of edges) {
        const type = (e.dependency_type as DependencyType) || e.reference_type || e.type || "generic_dependency";
        els.push({
            data: {
                id: `${e.source}__${e.target}__${type}`,
                source: e.source,
                target: e.target,
                label: type,
                depType: type,
                raw: e,
            },
        });
    }

    return els;
}

export default function GraphView({ graph, onSelectNode, height = "70vh" }: Props) {
    const cyRef = useRef<Cytoscape.Core | null>(null);
    const [layoutName, setLayoutName] = useState<"dagre" | "cose">("dagre");

    const elements = useMemo(() => toElements(graph.nodes, graph.edges), [graph]);

    // Re-run layout when layoutName or elements change
    useEffect(() => {
        const cy = cyRef.current;
        if (!cy) return;
        const layout = cy.layout(
            layoutName === "dagre"
                ? ({ name: "dagre", rankDir: "TB", nodeSep: 30, rankSep: 60, edgeSep: 20 } as any)
                : ({ name: "cose", nodeOverlap: 10, idealEdgeLength: 120, edgeElasticity: 0.2 } as any)
        );
        layout.run();
    }, [layoutName, elements]);

    const stylesheet = useMemo(
        () => [
            {
                selector: "node",
                style: {
                    "background-color": (ele: any) => {
                        const t = ele.data("type") as string;
                        return NODE_COLORS[t] ?? NODE_COLORS.unknown;
                    },
                    label: "data(label)",
                    color: "#111827",
                    "font-family": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial",
                    "font-weight": 600,
                    "font-size": "10px",
                    "text-wrap": "wrap",
                    "text-max-width": 120,
                    "text-halign": "center",
                    "text-valign": "bottom",
                },
            },
            {
                selector: "edge",
                style: {
                    width: 2,
                    "line-color": (ele: any) => edgeColor(ele.data("depType")),
                    "target-arrow-color": (ele: any) => edgeColor(ele.data("depType")),
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "arrow-scale": 1,
                },
            },
            {
                selector: "node:selected",
                style: {
                    "border-width": 3,
                    "border-color": "#111827",
                },
            },
            {
                selector: "edge:selected",
                style: {
                    width: 4,
                },
            },
        ],
        []
    );

    return (
        <div className="w-full">
            <div className="flex items-center gap-2 mb-2">
                <span className="text-sm text-gray-600">
                    Nodes: {graph.stats?.node_count ?? graph.nodes.length} · Edges:{" "}
                    {graph.stats?.edge_count ?? graph.edges.length}
                </span>
                <div className="ml-auto flex items-center gap-2">
                    <label className="text-sm">Layout:</label>
                    <select
                        className="border rounded px-2 py-1 text-sm"
                        value={layoutName}
                        onChange={(e) => setLayoutName(e.target.value as any)}
                    >
                        <option value="dagre">Hierarchical (dagre)</option>
                        <option value="cose">Force-directed (cose)</option>
                    </select>
                    <button
                        className="border rounded px-2 py-1 text-sm"
                        onClick={() => {
                            const cy = cyRef.current;
                            if (!cy) return;
                            cy.fit(undefined, 30);
                        }}
                    >
                        Center
                    </button>
                </div>
            </div>
            <CytoscapeComponent
                elements={elements}
                cy={(cy: any) => {
                    cyRef.current = cy;
                    cy.on("tap", "node", (evt: any) => {
                        const data = evt.target.data("raw") as ArtifactNode;
                        onSelectNode?.(data);
                    });
                    cy.on("tap", (evt: any) => {
                        // click on background
                        if (evt.target === cy) onSelectNode?.(null);
                    });
                    // initial fit
                    setTimeout(() => cy.fit(undefined, 30), 50);
                }}
                style={{ width: "100%", height }}
                stylesheet={stylesheet as any}
                layout={{ name: layoutName }}
            />
        </div>
    );
}
