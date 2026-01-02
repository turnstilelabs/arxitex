import { edgeKey } from './data';

function cleanLatexForDisplay(content?: string | null): string {
  if (!content) return '';
  return String(content).replace(/\\label\{[^}]*\}/g, '').trim();
}

function normalizeTextForDedupe(str?: string | null): string {
  if (!str) return '';
  return cleanLatexForDisplay(str)
    .replace(/\\label\{[^}]*\}/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function topoSort(nodeIds: string[], edges: Array<[string, string]>): string[] {
  const adj = new Map<string, string[]>(nodeIds.map((id) => [id, []]));
  const indeg = new Map<string, number>(nodeIds.map((id) => [id, 0]));

  edges.forEach(([u, v]) => {
    if (!adj.has(u) || !indeg.has(v)) return;
    adj.get(u)!.push(v);
    indeg.set(v, (indeg.get(v) || 0) + 1);
  });

  const queue: string[] = [];
  indeg.forEach((deg, id) => {
    if (deg === 0) queue.push(id);
  });

  const order: string[] = [];
  while (queue.length) {
    const u = queue.shift()!;
    order.push(u);
    for (const v of adj.get(u) || []) {
      indeg.set(v, (indeg.get(v) || 0) - 1);
      if ((indeg.get(v) || 0) === 0) queue.push(v);
    }
  }

  if (order.length !== nodeIds.length) {
    const seen = new Set(order);
    for (const id of nodeIds) if (!seen.has(id)) order.push(id);
  }

  return order;
}

function splitPrereqItems(text?: string | null): string[] {
  if (!text) return [];
  const t = String(text).replace(/\r\n/g, '\n').trim();
  if (!t) return [];

  const paragraphs = t.split(/\n\s*\n+/);
  const items: string[] = [];

  paragraphs.forEach((para) => {
    const lines = para.split('\n');
    let current = '';

    const pushCurrent = () => {
      const cleaned = cleanLatexForDisplay(current).replace(/\s+\n/g, '\n').trim();
      if (cleaned) items.push(cleaned);
      current = '';
    };

    for (const line of lines) {
      const isHeaderish = /^\s*(?:\$[^$]{0,80}\$|[A-Za-z\\][^:\n\r]{0,80}):/.test(line);
      if (isHeaderish && current.trim()) {
        pushCurrent();
        current = line;
      } else {
        current = current ? `${current}\n${line}` : line;
      }
    }
    pushCurrent();
  });

  return items;
}

export type DistillSupporting = {
  id: string;
  title: string;
  content: string;
};

export type DistillModel = {
  title: string;
  depth: number;
  target: {
    id: string;
    title: string;
    content: string;
  };
  definitions: string[];
  supporting: DistillSupporting[];
  adj: Record<string, string[]>;
  graphNodes: Record<string, { id: string; title: string; content: string }>;
  tree: { id: string; children: any[] };
} | null;

export function buildDistillModel(
  state: any,
  nodeById: Map<string, any>,
  incomingEdgesByTarget: Map<string, Array<{ s: string; t: string; dep: string }>>,
  graphData: { nodes: any[]; edges: any[] },
): DistillModel {
  if (!state.proofMode || !state.proofTargetId) return null;

  const excludeTypes = new Set(['remark', 'unknown']);
  const allowedResultTypes = new Set(['theorem', 'lemma', 'proposition', 'corollary', 'claim']);

  const allIds = Array.from(state.proofVisibleNodes || []);
  const keptIds = allIds.filter((id) => {
    const n = nodeById.get(id);
    if (!n) return false;
    if (excludeTypes.has(n.type)) return false;
    return true;
  });

  const keptSet = new Set<string>(keptIds);

  const edges: Array<[string, string]> = [];
  graphData.edges.forEach((e: any) => {
    const sId = typeof e.source === 'object' ? e.source.id : e.source;
    const tId = typeof e.target === 'object' ? e.target.id : e.target;
    if (!keptSet.has(sId) || !keptSet.has(tId)) return;
    const key = edgeKey(sId, tId);
    if (!state.proofVisibleEdges.has(key)) return;
    const dep = e.dependency_type || 'internal';
    if (dep !== 'used_in') return;
    edges.push([sId, tId]);
  });

  const topo = topoSort(keptIds, edges);

  const defSet = new Set<string>();
  const definitions: string[] = [];
  for (const id of topo) {
    const n = nodeById.get(id);
    const items = splitPrereqItems((n && (n.prerequisites_preview || n.prerequisites)) || '');
    for (const item of items) {
      const key = normalizeTextForDedupe(item);
      if (key && !defSet.has(key)) {
        defSet.add(key);
        definitions.push(item);
      }
    }
  }

  const supporting: DistillSupporting[] = [];
  for (const id of topo) {
    if (id === state.proofTargetId) continue;
    const n = nodeById.get(id);
    if (!n || !allowedResultTypes.has(n.type)) continue;
    const content = cleanLatexForDisplay(n.content_preview || n.content || '').trim();
    if (!content) continue;
    supporting.push({
      id,
      title: n.display_name || n.label || n.id,
      content,
    });
  }

  const target = nodeById.get(state.proofTargetId) || {};

  const adj: Record<string, string[]> = {};
  const graphNodes: Record<string, { id: string; title: string; content: string }> = {};

  (function buildAdj(startId: string) {
    const visited = new Set<string>([startId]);
    const queue: string[] = [startId];

    while (queue.length) {
      const id = queue.shift()!;
      const ins = incomingEdgesByTarget.get(id) || [];
      for (const { s, dep } of ins) {
        if (dep === 'generalized_by') continue;
        const n = nodeById.get(s);
        if (!n || !allowedResultTypes.has(n.type)) continue;
        if (!adj[id]) adj[id] = [];
        if (!adj[id].includes(s)) adj[id].push(s);
        if (!visited.has(s)) {
          visited.add(s);
          queue.push(s);
        }
      }
    }

    visited.forEach((id) => {
      const n = nodeById.get(id);
      if (!n) return;
      const content = cleanLatexForDisplay(n.content_preview || n.content || '').trim();
      graphNodes[id] = {
        id,
        title: n.display_name || n.label || n.id,
        content,
      };
    });
  })(state.proofTargetId as string);

  const childrenMap = new Map<string, string[]>();
  graphData.edges.forEach((e: any) => {
    const sId = typeof e.source === 'object' ? e.source.id : e.source;
    const tId = typeof e.target === 'object' ? e.target.id : e.target;
    if (!keptSet.has(sId) || !keptSet.has(tId)) return;
    const key = edgeKey(sId, tId);
    const dep = e.dependency_type || 'internal';
    if (dep !== 'used_in') return;
    if (!state.proofVisibleEdges.has(key)) return;
    if (!childrenMap.has(tId)) childrenMap.set(tId, []);
    childrenMap.get(tId)!.push(sId);
  });

  function buildTree(id: string, level: number): { id: string; children: any[] } {
    if (level >= state.proofDepth) return { id, children: [] };
    const kids = (childrenMap.get(id) || []).map((cid) => buildTree(cid, level + 1));
    return { id, children: kids };
  }

  const tree = buildTree(state.proofTargetId as string, 0);

  return {
    title: target.display_name || target.label || String(state.proofTargetId),
    depth: state.proofDepth,
    target: {
      id: state.proofTargetId,
      title: target.display_name || target.label || String(state.proofTargetId),
      content: cleanLatexForDisplay(target.content_preview || target.content || '').trim(),
    },
    definitions,
    supporting,
    adj,
    graphNodes,
    tree,
  };
}

export function buildDistilledHtml(model: DistillModel): string {
  if (!model) {
    return '<!DOCTYPE html><html><body><p>No distilled model available.</p></body></html>';
  }

  const { title, target, definitions, supporting } = model;
  const graphNodesModel = model.graphNodes || {};
  const tree = model.tree;

  function renderTreeNode(node: any): string {
    const info = graphNodesModel[node.id] || { id: node.id, title: node.id, content: '' };
    const childrenHtml = (node.children || []).map(renderTreeNode).join('\n');
    return `
            <div class="result-item" data-id="${info.id}">
                <h3>${info.title} <span class="artifact-controls" data-id="${info.id}">
                  <button class="fold-btn" data-act="less" title="Unfold less">−</button>
                  <button class="fold-btn" data-act="more" title="Unfold more">+</button>
                </span></h3>
                <div class="math-content">${info.content || '<em>No statement available.</em>'}</div>
                <div class="child-results" data-parent="${info.id}">${childrenHtml}</div>
            </div>`;
  }

  const defsSection = definitions && definitions.length
    ? definitions.map((d) => `<div class="def-item"><div class="math-content">${d}</div></div>`).join('\n')
    : '<p class="muted">No explicit definitions or notations were required beyond the visible path.</p>';

  const suppSection = (tree && tree.children && tree.children.length)
    ? tree.children.map(renderTreeNode).join('\n')
    : (supporting && supporting.length
      ? supporting.map((r) => `
            <div class="result-item" data-id="${r.id}">
                <h3>${r.title} <span class="artifact-controls" data-id="${r.id}">
                  <button class="fold-btn" data-act="less" title="Unfold less">−</button>
                  <button class="fold-btn" data-act="more" title="Unfold more">+</button>
                </span></h3>
                <div class="math-content">${r.content}</div>
                <div class="child-results" data-parent="${r.id}"></div>
            </div>`).join('\n')
      : '<p class="muted">No intermediate results are required at the current unfolding depth.</p>');

  const targetSection = `
        <div class="target-item">
            <h2>${target.title}</h2>
            <div class="math-content">${target.content || '<em>No statement available.</em>'}</div>
        </div>`;

  const bootSrc = '/constellations/distilled_boot.js';
  const explainerSrc = '/constellations/explainer.js';
  const themeCss = '/constellations/style.css';

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Distilled Proof for: ${title}</title>

<link rel="stylesheet" href="${themeCss}" />

<style>
  body { padding: 0 !important; }

  .doc {
    max-width: 980px;
    margin: 20px auto;
    padding: 0 18px 60px;
  }

  .dp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: var(--surface1);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 18px;
  }

  .dp-header h1 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 18px;
    margin: 0;
    color: var(--accent);
  }

  .dp-actions { display: flex; gap: 10px; flex-wrap: wrap; }

  .dp-actions .depth-btn { display: inline-flex; align-items: center; gap: 6px; }
  .dp-actions .icon svg { width: 18px; height: 18px; display: block; }

  section {
    margin: 18px 0;
    background: var(--surface1);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px 16px;
  }

  section h2 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 16px;
    margin: 0 0 10px;
    color: var(--accent);
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
  }

  h3 {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 15px;
    margin: 0 0 6px;
    color: var(--primary-text);
  }

  .muted { color: var(--secondary-text); font-style: italic; }
  .result-item, .def-item, .target-item { margin: 12px 0; }
  .math-content { line-height: 1.65; color: var(--primary-text); }

  .math-content, .math-content * { -webkit-user-select: text; user-select: text; }
  mjx-container, mjx-container * { -webkit-user-select: text; user-select: text; }

  .result-item .child-results {
    margin-left: 16px;
    border-left: 2px solid var(--border-color);
    padding-left: 12px;
  }
  .result-item.collapsed > .child-results { display: none; }
  .artifact-controls { display: inline-flex; gap: 6px; margin-left: 8px; vertical-align: middle; }

  .fold-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--surface2);
    cursor: pointer;
    font-weight: 800;
    line-height: 1;
    color: var(--primary-text);
    font-family: 'Inter', system-ui, sans-serif;
  }
  .fold-btn:hover { background: var(--surface1); }

  .explainer-menu {
    position: fixed;
    display: none;
    background: var(--surface2);
    color: var(--primary-text);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    padding: 6px;
    z-index: 2000;
    min-width: 220px;
  }
  .explainer-menu button {
    display: block;
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    padding: 8px 10px;
    cursor: pointer;
    border-radius: 6px;
    color: var(--primary-text);
    font-family: 'Inter', system-ui, sans-serif;
  }
  .explainer-menu button:hover {
    background: var(--surface1);
    border-color: var(--border-color);
  }

  .explainer-card {
    background: var(--surface2);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
  }
  .explainer-card header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 6px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 6px;
  }
  .explainer-card header .actions button { margin-left: 6px; font-size: 12px; }

  .key-modal {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.55);
    align-items: center;
    justify-content: center;
    z-index: 2001;
  }
  .key-modal .dialog {
    background: var(--surface1);
    color: var(--primary-text);
    width: 380px;
    max-width: 92vw;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid var(--border-color);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
  }
  .key-modal .dialog h3 {
    margin: 0 0 10px;
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 15px;
    color: var(--accent);
  }
  .key-modal .dialog input {
    width: 100%;
    padding: 10px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--surface2);
    color: var(--primary-text);
  }
  .key-modal .dialog .actions {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
    margin-top: 10px;
  }

  .overlay-layer { position: fixed; inset: 0; pointer-events: none; z-index: 1500; }
  .highlight-rect { position: fixed; background: rgba(255, 221, 87, 0.35); outline: 1px solid #ffd94a; border-radius: 2px; }
  #pin-layer { position: fixed; inset: 0; pointer-events: none; z-index: 1501; }
  .pin { position: fixed; width: 18px; height: 18px; border-radius: 50%; background: #ffd94a; border: 1px solid #caa500; box-shadow: 0 1px 2px rgba(0,0,0,0.2); pointer-events: auto; cursor: pointer; }

  .comment-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 360px;
    height: 100vh;
    background: var(--surface2);
    color: var(--primary-text);
    border-left: 1px solid var(--border-color);
    transform: translateX(100%);
    transition: transform 200ms ease;
    z-index: 1600;
    display: flex;
    flex-direction: column;
  }
  .comment-panel.open { transform: translateX(0); }
  .comment-panel header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-color);
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 700;
    color: var(--accent);
  }
  .comment-panel header .close {
    font-size: 22px;
    border: none;
    background: none;
    cursor: pointer;
    color: var(--secondary-text);
  }
  .comment-panel header .close:hover { color: var(--primary-text); }

  .threads { overflow: auto; padding: 10px; flex: 1; }
  .thread-card {
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 10px;
    background: var(--surface1);
  }
  .thread-card.active { outline: 2px solid var(--accent); }
  .thread-card header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'Inter', system-ui, sans-serif;
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 6px;
    color: var(--primary-text);
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 6px;
  }
  .thread-card header .actions .icon-btn {
    width: 24px;
    height: 24px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 18px;
    line-height: 1;
    color: var(--secondary-text);
    border-radius: 4px;
  }
  .thread-card header .actions .icon-btn:hover { background: var(--surface2); color: var(--primary-text); }
  .thread-card .content { max-height: 320px; overflow: auto; }

  @media print {
    .dp-actions { display: none; }
    a { color: inherit; text-decoration: none; }
  }
</style>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\\(', '\\\)']],
      displayMath: [['$$', '$$'], ['\\\[', '\\\]']],
      packages: {'[+]': ['ams']},
      macros: { bbE: '\\mathbb{E}', bbP: '\\mathbb{P}', bbZ: '\\mathbb{Z}', bbR: '\\mathbb{R}', bbG: '\\mathbb{G}', bbH: '\\mathbb{H}', bbV: '\\mathbb{V}', mathbbm: ['{\\mathbf{#1}}', 1], mathbbm1: '\\mathbf{1}', llbracket: '\\mathopen{[\\![}', rrbracket: '\\mathclose{]\\!]}' }
    },
    options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
</head>
<body class="visualization-page">
<div class="doc">
  <header class="dp-header">
    <h1>Distilled Proof: ${title}</h1>
    <div class="dp-actions">
      <button id="download-tex" class="depth-btn depth-btn--primary">
        <span class="icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M7.5 10.5l4.5 4.5 4.5-4.5M12 3v12"/></svg></span>
        Download LaTeX
      </button>
    </div>
  </header>

  <section>
    <h2>Definitions and Notations</h2>
    ${defsSection}
  </section>

  <section>
    <h2>Supporting Results</h2>
    ${suppSection}
  </section>

  <section>
    <h2>Target Theorem and Proof</h2>
    ${targetSection}
  </section>
</div>

<script type="application/json" id="distill-data">${JSON.stringify(model)}</script>
<div id="explainer-menu" class="explainer-menu"></div>
<div id="highlight-layer" class="overlay-layer"></div>
<div id="pin-layer"></div>
<aside id="comment-panel" class="comment-panel" aria-label="Comments">
  <header>
    <div>Comments</div>
    <button id="close-panel" class="close" title="Close">×</button>
  </header>
  <div id="threads" class="threads"></div>
</aside>
<div id="key-modal" class="key-modal">
  <div class="dialog">
    <h3>Set OpenAI API key</h3>
    <input id="key-input" type="password" placeholder="sk-..." />
    <div class="actions">
      <button id="key-cancel" class="depth-btn">Cancel</button>
      <button id="key-save" class="depth-btn depth-btn--primary">Save</button>
    </div>
  </div>
</div>
<script src="${bootSrc}"></script>
<script src="${explainerSrc}"></script>
</body>
</html>`;
  return html;
}

export function renderDistilledWindow(model: DistillModel) {
  if (!model) return;

  // Store the model in a tab-persistent location that survives navigation.
  // `window.name` has a much higher practical size limit than localStorage and
  // avoids quota issues for large distilled models.
  try {
    window.name = JSON.stringify({ __arxigraphDistilledModel: model });
  } catch {
    // ignore
  }

  // Persist model so the dedicated distilled route can read it client-side.
  try {
    window.localStorage.setItem('arxigraph:distilledModel', JSON.stringify(model));
  } catch {
    // Ignore storage errors; the distilled route will handle missing data.
  }

  // Navigate within the app so the browser Back button naturally returns to the graph page.
  const basePath = window.location.pathname.replace(/\/+$/, '');
  const targetUrl = `${basePath}/distilled`;
  window.location.assign(targetUrl);
}
