const loadStatus = document.getElementById("loadStatus");
const querySelect = document.getElementById("querySelect");
const methodSelect = document.getElementById("methodSelect");
const queryMeta = document.getElementById("queryMeta");
const queryRefs = document.getElementById("queryRefs");
const resultsGrid = document.getElementById("resultsGrid");

const state = {
  graph: null,
  queries: [],
  results: {
    e1: null,
    e2: null,
    e3: null,
    e4: null,
    e5: null,
  },
  qrelsIndex: {},
  nodeIndex: {},
  methods: [
    { key: "e1", label: "BM25 (e1)", field: "artifact_ids" },
    { key: "e2", label: "Dense (e2)", field: "artifact_ids" },
    { key: "e3", label: "PyLate (e3)", field: "artifact_ids" },
    { key: "e4", label: "Graph‑Expand (e4)", field: "expanded_ids" },
    { key: "e5", label: "Hybrid RRF (e5)", field: "artifact_ids" },
  ],
};

const DEFAULT_PATHS = {
  graph: "/data/graphs/perfectoid.json",
  queries: "/data/citation_dataset/perfectoid_queries.jsonl",
  e1: "/data/retrieval/perfectoid/perfectoid_best_20260227T093235Z/e1_content+all/auto/e1_results.jsonl",
  e2: "/data/retrieval/perfectoid/perfectoid_best_20260227T093359Z/e2_content+semantic/auto/e2_results.jsonl",
  e3: "/data/retrieval/perfectoid/perfectoid_best_20260227T093622Z/e3_content+all/auto/e3_results.jsonl",
  e4: "",
  e5: "",
};

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

function parseJsonl(text) {
  const rows = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    rows.push(JSON.parse(trimmed));
  }
  return rows;
}

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status}`);
  }
  return res.text();
}

function buildNodeIndex(graph) {
  const nodes = graph.graph ? graph.graph.nodes : graph.nodes;
  const index = {};
  for (const node of nodes) {
    index[node.id] = node;
  }
  return index;
}

function inferLabelNumber(node) {
  if (!node) return "";
  if (node.pdf_label_number) return node.pdf_label_number;
  const label = node.pdf_label || "";
  const m = label.match(/(\d+(?:\.\d+)*)/);
  return m ? m[1] : "";
}

function buildQrelsIndex(graph, queries) {
  const nodes = graph.graph ? graph.graph.nodes : graph.nodes;
  const labelIndex = {};
  for (const node of nodes) {
    const kind = (node.type || "").toLowerCase().replace(".", "");
    const number = inferLabelNumber(node);
    if (!kind || !number) continue;
    const key = `${kind}:${number}`;
    if (!labelIndex[key]) labelIndex[key] = [];
    labelIndex[key].push(node.id);
  }

  const out = {};
  for (const q of queries) {
    const refs = q.explicit_refs || [];
    const ids = [];
    for (const ref of refs) {
      const kind = (ref.kind || "").toLowerCase().replace(".", "");
      const number = (ref.number || "").trim();
      if (!kind || !number) continue;
      const key = `${kind}:${number}`;
      const hit = labelIndex[key] || [];
      ids.push(...hit);
    }
    if (ids.length) {
      out[q.query_id] = Array.from(new Set(ids));
    }
  }
  return out;
}

function _balanceMathDelimiters(text) {
  if (!text) return text;
  const dollarMatches = text.match(/(?<!\\)\$/g) || [];
  if (dollarMatches.length % 2 === 1) {
    const last = text.lastIndexOf("$");
    if (last !== -1) {
      text = text.slice(0, last);
    }
  }
  const openParen = text.lastIndexOf("\\(");
  const closeParen = text.lastIndexOf("\\)");
  if (openParen > closeParen) {
    text = text.slice(0, openParen);
  }
  const openBrack = text.lastIndexOf("\\[");
  const closeBrack = text.lastIndexOf("\\]");
  if (openBrack > closeBrack) {
    text = text.slice(0, openBrack);
  }
  return text;
}

function truncate(text, maxLen = 420) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  const slice = text.slice(0, maxLen);
  const lastStop = Math.max(slice.lastIndexOf("."), slice.lastIndexOf(";"), slice.lastIndexOf(":"));
  const cut = lastStop > 80 ? slice.slice(0, lastStop + 1) : slice;
  const balanced = _balanceMathDelimiters(cut);
  return balanced.trim() + "...";
}

function getNodeForArtifact(artifactId) {
  if (!artifactId) return null;
  const baseId = artifactId.replace(/#proof$/, "");
  return state.nodeIndex[baseId] || null;
}

function artifactSnippet(artifactId) {
  const node = getNodeForArtifact(artifactId);
  if (!node) return "";
  if (artifactId.endsWith("#proof")) {
    return truncate(node.proof || "");
  }
  return truncate(node.content || "");
}

function artifactFullText(artifactId) {
  const node = getNodeForArtifact(artifactId);
  if (!node) return "";
  if (artifactId.endsWith("#proof")) {
    return node.proof || "";
  }
  return node.content || "";
}

function _sanitizeNonMath(text) {
  if (!text) return "";
  // Strip common LaTeX commands in non-math text segments.
  let s = text;
  s = s.replace(/\\label\{[^}]*\}/g, "");
  s = s.replace(/\\ref\{[^}]*\}/g, "");
  s = s.replace(/\\cite\{[^}]*\}/g, "");
  s = s.replace(/\\begin\{[^}]*\}/g, "");
  s = s.replace(/\\end\{[^}]*\}/g, "");
  s = s.replace(/\\(textbf|textit|emph)\{([^}]*)\}/g, "$2");
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

function _sanitizeForDisplay(text) {
  if (!text) return "";
  const parts = [];
  const regex = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$(?:\\\$|[^\$])+\$)/g;
  let last = 0;
  let m;
  while ((m = regex.exec(text)) !== null) {
    const before = text.slice(last, m.index);
    if (before) parts.push(_sanitizeNonMath(before));
    parts.push(m[0]); // keep math untouched
    last = m.index + m[0].length;
  }
  const tail = text.slice(last);
  if (tail) parts.push(_sanitizeNonMath(tail));
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

function renderQueryOptions() {
  querySelect.innerHTML = "";
  for (const q of state.queries) {
    const opt = document.createElement("option");
    opt.value = q.query_id;
    opt.textContent = `${q.query_text} (${q.query_style || "?"})`;
    querySelect.appendChild(opt);
  }
}

function renderQuery(qid) {
  const q = state.queries.find((row) => row.query_id === qid);
  if (!q) return;
  const refs = q.explicit_refs || [];
  const refText = refs.length
    ? refs.map((r) => `${r.kind || "?"} ${r.number || "?"}`).join(", ")
    : "none";
  const relIds = state.qrelsIndex[qid] || [];
  queryMeta.textContent = q.query_text;
  queryRefs.textContent = `Explicit refs: ${refText} • Qrels: ${relIds.length}`;
  renderResults(qid, relIds, methodSelect.value);
  renderMath(queryMeta);
}

function renderResults(qid, relIds, methodKey) {
  resultsGrid.innerHTML = "";
  const method = state.methods.find((m) => m.key === methodKey);
  if (!method) return;

  const col = document.createElement("div");
  col.className = "panel";
  const title = document.createElement("div");
  title.className = "method-title";
  title.textContent = method.label;
  col.appendChild(title);

  const results = state.results[method.key] || {};
  const row = results[qid] || {};
  const ids = row[method.field] || row.artifact_ids || [];
  const scores = row.scores || [];

  const firstRelRank = (() => {
    for (let i = 0; i < ids.length; i += 1) {
      const baseId = ids[i].replace(/#proof$/, "");
      if (relIds.includes(baseId)) return i + 1;
    }
    return null;
  })();
  const hit10 = firstRelRank !== null && firstRelRank <= 10;
  const meta = document.createElement("div");
  meta.className = "method-meta";
  meta.innerHTML = `
    <span class="badge ${hit10 ? "ok" : "muted"}">${hit10 ? "Hit@10" : "Miss@10"}</span>
    <span class="muted">First relevant: ${firstRelRank || "—"}</span>
  `;
  col.appendChild(meta);
  const list = document.createElement("div");
  list.className = "results";

  if (!ids.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No results.";
    list.appendChild(empty);
  } else {
    ids.forEach((id, idx) => {
      const node = getNodeForArtifact(id);
      const item = document.createElement("div");
      const isRel = relIds.includes(id.replace(/#proof$/, ""));
      item.className = `result-item${isRel ? " relevant" : ""}`;
      const label = node && node.pdf_label ? node.pdf_label : "";
      const type = node && node.type ? node.type : "unknown";
      const score = typeof scores[idx] === "number" ? scores[idx].toFixed(4) : null;
      const fullText = artifactFullText(id);
      const hasFull = fullText && fullText.length > 0;
      const fullId = `full-${method.key}-${qid}-${idx}`.replace(/[^a-zA-Z0-9_-]/g, "");
      item.innerHTML = `
        <span class="badge">${idx + 1}</span>
        <span class="badge">${type}</span>
        ${label ? `<span class="badge">${label}</span>` : ""}
        ${score !== null ? `<span class="badge score">score ${score}</span>` : ""}
        <span class="rel-label ${isRel ? "ok" : "muted"}">${isRel ? "relevant" : "not relevant"}</span>
        <div class="artifact-id"></div>
      `;
      const idDiv = item.querySelector(".artifact-id");
      if (idDiv) idDiv.textContent = id;

      const snippetDiv = document.createElement("div");
      snippetDiv.className = "muted snippet";
      snippetDiv.textContent = _sanitizeForDisplay(artifactSnippet(id));
      item.appendChild(snippetDiv);

      if (hasFull) {
        const btn = document.createElement("button");
        btn.className = "expand-btn";
        btn.type = "button";
        btn.setAttribute("data-target", fullId);
        btn.textContent = "Show full";
        item.appendChild(btn);

        const fullDiv = document.createElement("div");
        fullDiv.id = fullId;
        fullDiv.className = "full-text";
        fullDiv.hidden = true;
        fullDiv.textContent = _sanitizeForDisplay(fullText);
        item.appendChild(fullDiv);
      }
      list.appendChild(item);
    });
  }
  col.appendChild(list);
  resultsGrid.appendChild(col);
  renderMath(col);
}

function renderMath(root) {
  if (!window.renderMathInElement || !root) return;
  try {
    window.renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  } catch (err) {
    console.warn("Math render failed", err);
  }
}

function renderMethodOptions() {
  methodSelect.innerHTML = "";
  for (const method of state.methods) {
    if (!state.results[method.key]) continue;
    const opt = document.createElement("option");
    opt.value = method.key;
    opt.textContent = method.label;
    methodSelect.appendChild(opt);
  }
  if (!methodSelect.value && methodSelect.options.length) {
    methodSelect.value = methodSelect.options[0].value;
  }
}

async function loadFromPaths() {
  loadStatus.textContent = "Loading…";
  const graphPath = DEFAULT_PATHS.graph;
  const queriesPath = DEFAULT_PATHS.queries;
  if (!graphPath || !queriesPath) {
    loadStatus.textContent = "Graph + queries paths are required.";
    return;
  }

  const graphText = await fetchText(graphPath);
  state.graph = JSON.parse(graphText);
  state.nodeIndex = buildNodeIndex(state.graph);

  const queriesText = await fetchText(queriesPath);
  state.queries = parseJsonl(queriesText);

  const pathMap = {
    e1: DEFAULT_PATHS.e1,
    e2: DEFAULT_PATHS.e2,
    e3: DEFAULT_PATHS.e3,
    e4: DEFAULT_PATHS.e4,
    e5: DEFAULT_PATHS.e5,
  };
  for (const key of Object.keys(pathMap)) {
    const path = pathMap[key];
    if (!path) continue;
    const text = await fetchText(path);
    const rows = parseJsonl(text);
    const index = {};
    for (const row of rows) {
      if (row.query_id) index[row.query_id] = row;
    }
    state.results[key] = index;
  }

  state.qrelsIndex = buildQrelsIndex(state.graph, state.queries);
  renderQueryOptions();
  renderMethodOptions();
  if (state.queries.length) {
    renderQuery(state.queries[0].query_id);
  }
  loadStatus.textContent = `Loaded ${state.queries.length} queries.`;
}

querySelect.addEventListener("change", (e) => {
  renderQuery(e.target.value);
});

methodSelect.addEventListener("change", (e) => {
  const qid = querySelect.value;
  if (!qid) return;
  const relIds = state.qrelsIndex[qid] || [];
  renderResults(qid, relIds, e.target.value);
});

function canFetch() {
  return window.location.protocol === "http:" || window.location.protocol === "https:";
}

document.addEventListener("DOMContentLoaded", () => {
  if (!canFetch()) {
    loadStatus.textContent = "Open via http:// to auto-load paths (file:// blocks fetch).";
    return;
  }
  loadFromPaths().catch((err) => {
    console.error(err);
    loadStatus.textContent = `Auto-load failed: ${err.message || err}`;
  });
});

document.addEventListener("click", (e) => {
  const btn = e.target.closest(".expand-btn");
  if (!btn) return;
  const targetId = btn.getAttribute("data-target");
  const full = document.getElementById(targetId);
  if (!full) return;
  const willShow = full.hasAttribute("hidden");
  if (willShow) {
    full.removeAttribute("hidden");
    btn.textContent = "Hide full";
    renderMath(full);
  } else {
    full.setAttribute("hidden", "true");
    btn.textContent = "Show full";
  }
});
