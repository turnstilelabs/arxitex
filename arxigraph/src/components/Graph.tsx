'use client';

import { forwardRef } from 'react';

import ConstellationsGraph, { type ConstellationsGraphHandle } from '@/components/ConstellationsGraph';
import type { ConstellationEdge, ConstellationNode } from '@/components/constellations/types';

interface GraphProps {
  nodes?: ConstellationNode[];
  links?: ConstellationEdge[];
  onReportNode?: (node: { id: string; label: string; type?: string }) => void;
  stats?: { artifacts: number; links: number };
  onReportGraph?: () => void;
}

const Graph = forwardRef<ConstellationsGraphHandle, GraphProps>(function Graph(
  { nodes = [], links = [], onReportNode, stats, onReportGraph },
  ref,
) {
  return (
    <ConstellationsGraph
      ref={ref}
      nodes={nodes}
      links={links}
      onReportNode={onReportNode}
      stats={stats}
      onReportGraph={onReportGraph}
    />
  );
});

export default Graph;
export type { ConstellationsGraphHandle };
