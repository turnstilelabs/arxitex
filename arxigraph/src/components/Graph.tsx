'use client';

import { forwardRef } from 'react';

import ConstellationsGraph, { type ConstellationsGraphHandle } from '@/components/ConstellationsGraph';
import type { ConstellationEdge, ConstellationNode } from '@/components/constellations/types';

interface GraphProps {
  nodes?: ConstellationNode[];
  links?: ConstellationEdge[];
}

const Graph = forwardRef<ConstellationsGraphHandle, GraphProps>(function Graph(
  { nodes = [], links = [] },
  ref,
) {
  return <ConstellationsGraph ref={ref} nodes={nodes} links={links} />;
});

export default Graph;
export type { ConstellationsGraphHandle };
