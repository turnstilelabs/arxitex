import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from loguru import logger


def create_visualization_html(graph_data: Dict, output_path: Path) -> None:
    """
    Creates an HTML file with an interactive D3.js graph visualization.

    This version is enhanced to:
    - Robustly handle multiple levels of backslash escaping in the source JSON
      by repeatedly un-escaping until the LaTeX is clean.
    - Use MathJax to render LaTeX formulas in tooltips by safely injecting
      math content as text, preventing conflicts with the HTML parser.
    - Correctly display the specific dependency type on edges.
    - Display both the content preview and prerequisite definitions for nodes.
    - Include improved styling for better readability.

    Args:
        graph_data: A dictionary containing the graph data (nodes, edges) and stats.
        output_path: The path where the HTML file will be saved.
    """
    html_template = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv Paper Dependency Graph - __ARXIV_ID__</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>

    <!-- MATHJAX INTEGRATION -->
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']],
          displayMath: [['$$', '$$'], ['\\[', '\\]']],
          processEscapes: true,
          packages: {'[+]': ['ams']},
          macros: {
            bbE: '\\mathbb{E}',
            bbP: '\\mathbb{P}',
            bbZ: '\\mathbb{Z}',
            bbR: '\\mathbb{R}',
            bbG: '\\mathbb{G}',
            bbH: '\\mathbb{H}',
            bbV: '\\mathbb{V}',
            mathbbm: ['{\\mathbf{#1}}', 1],
            mathbbm1: '\\mathbf{1}',
            llbracket: '\\mathopen{[\\![}',
            rrbracket: '\\mathclose{]\\!]}'
          }
        },
        options: {
          skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
          ignoreHtmlClass: 'tex2jax_ignore'
        },
        startup: {
          pageReady: () => {
            return MathJax.startup.defaultPageReady();
          }
        }
      };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; color: #333; }
        .header { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { margin: 0 0 10px 0; }
        .stats { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px; }
        .stat { background: #e9ecef; padding: 8px 12px; border-radius: 4px; font-size: 14px; }
        .graph-container { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }
        .controls { padding: 15px; background: #f8f9fa; border-bottom: 1px solid #dee2e6; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .controls button { padding: 6px 12px; border: 1px solid #dee2e6; background: white; border-radius: 4px; cursor: pointer; font-size: 14px; transition: background-color 0.2s; }
        .controls button:hover { background: #e9ecef; }
        .legend { display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin-left: auto; }
        .legend-item { display: flex; align-items: center; gap: 5px; font-size: 12px; }
        .legend-color { width: 12px; height: 12px; border-radius: 50%; }
        #graph { width: 100%; height: 70vh; cursor: grab; }
        #graph:active { cursor: grabbing; }
        .node { stroke: #fff; stroke-width: 2px; cursor: pointer; }
        .node:hover { stroke: #000; }
        .link { stroke: #999; stroke-opacity: 0.6; stroke-width: 1.5px; transition: stroke-width 0.2s, stroke-opacity 0.2s; cursor: pointer; marker-end: url(#arrowhead); }
        .link:hover { stroke-width: 4px; stroke-opacity: 1; }
        .node-label { font-size: 12px; fill: #333; text-anchor: middle; pointer-events: none; font-weight: 500; }
        .tooltip { position: absolute; background: rgba(0, 0, 0, 0.9); color: white; padding: 12px; border-radius: 6px; font-size: 14px; pointer-events: none; max-width: 450px; z-index: 1000; line-height: 1.6; border: 1px solid #333; }
        .tooltip h4 { margin: 0 0 10px 0; font-size: 16px; border-bottom: 1px solid #555; padding-bottom: 8px; }
        .tooltip p { margin: 5px 0; }
        .tooltip .id-label { color: #aaa; font-family: monospace; font-size: 11px; }
        .tooltip .edge-type { font-weight: bold; color: #ffc107; text-transform: uppercase; }
        .tooltip .math-content { margin-top: 5px; background: #222; padding: 8px; border-radius: 4px; overflow-x: auto; max-height: 300px; overflow-y: auto; white-space: pre-wrap; }
        .tooltip .error-message { color: #ff6b6b; font-style: italic; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>arXiv Paper Dependency Graph</h1>
        <p><strong>Paper ID:</strong> __ARXIV_ID__ | <strong>Generated:</strong> __TIMESTAMP__</p>
        <div class="stats">
            <div class="stat"><strong>__NODE_COUNT__</strong> artifacts</div>
            <div class="stat"><strong>__EDGE_COUNT__</strong> references</div>
        </div>
    </div>
    <div class="graph-container">
        <div class="controls">
            <button id="play-pause">⏸️ Pause</button>
            <button id="reset">🔄 Reset</button>
            <button id="center">🎯 Center View</button>
            <div class="legend" id="legend-container"></div>
        </div>
        <svg id="graph"></svg>
    </div>
    <div class="tooltip" id="tooltip" style="display: none;"></div>
    <script>
        const graphData = __GRAPH_DATA_JSON__;
        const nodeTypes = [...new Set(graphData.nodes.map(d => d.type))];
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        const nodeColors = nodeTypes.reduce((acc, type) => {
            acc[type] = colorScale(type);
            return acc;
        }, {});

        const svg = d3.select("#graph");
        const width = svg.node().getBoundingClientRect().width;
        const height = svg.node().getBoundingClientRect().height;

        svg.append("defs").append("marker")
            .attr("id", "arrowhead").attr("viewBox", "-0 -5 10 10").attr("refX", 25)
            .attr("refY", 0).attr("orient", "auto").attr("markerWidth", 8).attr("markerHeight", 8)
            .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#999");

        const zoom = d3.zoom().scaleExtent([0.1, 5]).on("zoom", (event) => g.attr("transform", event.transform));
        svg.call(zoom);
        const g = svg.append("g");

        svg.on("click", () => {
            if (pinned) {
                pinned = false;
                pinnedNode = null;
                tooltip.style("display", "none");
            }
        });

        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(120))
            .force("charge", d3.forceManyBody().strength(-500))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));

        const link = g.append("g").selectAll("line")
            .data(graphData.edges).enter().append("line").attr("class", "link");

        const node = g.append("g").selectAll("circle")
            .data(graphData.nodes).enter().append("circle").attr("class", "node")
            .attr("r", 15).attr("fill", d => nodeColors[d.type] || '#ccc')
            .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));

        const label = g.append("g").selectAll("text")
            .data(graphData.nodes).enter().append("text").attr("class", "node-label")
            .attr("dy", 28).text(d => d.display_name);

        const tooltip = d3.select("#tooltip");
        let pinned = false;
        let pinnedNode = null;

        // Replace corrupted characters and sanitize HTML/LaTeX before MathJax
        function sanitizeCorruptions(s) {
            if (!s) return '';
            // CORRECTED: Repeatedly un-escape backslashes until no more double backslashes exist.
            // This robustly handles cases where the input data is escaped multiple times.
            while (s.includes('\\\\')) {
                s = s.replace(/\\\\/g, '\\');
            }

            // Convert HTML line breaks to newlines for better typesetting
            s = s.replace(/<br\s*\/>/gi, '\n').replace(/<br\s*>/gi, '\n');
            // Map common corrupted control chars to backslashes (observed in some JSONs)
            s = s.replace(/[\u007F\u001F\u009F]/g, '\\');
            // Remove remaining control chars except tabs/newlines
            s = s.replace(/[\u0000-\u0008\u000B-\u000C\u000E-\u001E\u0080-\u009E]/g, '');
            // Convert old-style {\em ...} to \emph{...}
            s = s.replace(/\{\\em\s+([^}]*)\}/g, '\\emph{$1}');
            return s;
        }

        // Helper function to clean LaTeX content for proper display
        function cleanLatexForDisplay(content) {
            if (!content) return '';
            let s = sanitizeCorruptions(content);
            return s
                // Remove LaTeX labels
                .replace(/\\label\{[^}]*\}/g, '')
                // Remove section commands
                .replace(/\\(sub)?(sub)?section\*?\{[^}]*\}/g, '')
                // Remove document structure commands
                .replace(/\\(chapter|part|paragraph|subparagraph)\*?\{[^}]*\}/g, '')
                // Remove \title, \author, \date commands
                .replace(/\\(title|author|date|maketitle)\*?\{[^}]*\}/g, '')
                .replace(/\\maketitle/g, '')
                // Remove comments (but not escaped %)
                .replace(/(?<!\\)%.*$/gm, '')
                // Remove \usepackage, \documentclass, etc. (conservative)
                .replace(/\\(usepackage|documentclass|begin\{document\}|end\{document\}).*$/gm, '')
                // Normalize excessive whitespace
                .replace(/\n{3,}/g, '\n\n')
                .trim();
        }

        // Detect if content already includes any math delimiters/environments
        function hasMathDelimiters(content) {
            if (!content) return false;
            const re = /(\$\$|\$|\\\(|\\\)|\\\[|\\\]|\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|cases|pmatrix)\}|\\end\{(equation\*?|align\*?|gather\*?|multline\*?|cases|pmatrix)\})/;
            return re.test(content);
        }

        // If no delimiters are present but the text looks like TeX, wrap it in inline math
        function wrapWithInlineMathIfNeeded(content) {
            if (!content) return '';
            // Process line by line to avoid wrapping large blocks that already contain math
            const lines = content.split(/\n/);
            const wrapped = lines.map(line => {
                if (!line) return line;
                if (hasMathDelimiters(line)) return line;
                const looksLikeTex = /(\\[A-Za-z]+|\\\{|\\\}|\^|_)/.test(line);
                return looksLikeTex ? `\\(${line}\\)` : line;
            });
            return wrapped.join('\n');
        }

        // Wrap bare TeX commands that are outside existing math segments ($$, $, \(\), \[\])
        function wrapTeXOutsideMath(content) {
            if (!content) return '';
            const parts = content.split(/(\$\$[^$]*\$\$|\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\])/);
            const rebuilt = parts.map(seg => {
                if (!seg) return '';
                if (seg.startsWith('$$') || seg.startsWith('$') || seg.startsWith('\\(') || seg.startsWith('\\[')) {
                    return seg; // already math
                }
                // For non-math segments, wrap per line if they look like TeX
                return wrapWithInlineMathIfNeeded(seg);
            }).join('');
            return rebuilt;
        }

        function renderNodeTooltip(event, d) {
            // 1. Clean and prepare content first
            const rawPreview = d.content_preview || '';
            const cleanedPreview = cleanLatexForDisplay(rawPreview);
            const finalPreview = wrapTeXOutsideMath(cleanedPreview);
            let prereqsHtml = '';
            let hasPrereqs = false;
            if (d.prerequisites_preview) {
                const rawPrereqs = d.prerequisites_preview || '';
                const cleanedPrereqs = cleanLatexForDisplay(rawPrereqs);
                const finalPrereqs = wrapTeXOutsideMath(cleanedPrereqs);
                prereqsHtml = `<h4>Prerequisites</h4><div class="math-content math-prereq"></div>`;
                hasPrereqs = true;
            }

            // 2. Set complete HTML with containers; fill math via text() to preserve backslashes
            tooltip
                .style("display", "block")
                .html(`<h4>${d.display_name}</h4>
                       <p><span class="id-label">ID: ${d.id} | Label: ${d.label || 'N/A'}</span></p>
                       <div><strong>Preview:</strong></div>
                       <div class="math-content math-preview"></div>
                       ${prereqsHtml}`)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 28) + "px");

            // Fill math text content to avoid HTML interfering with backslashes
            tooltip.select('.math-preview').text(finalPreview);
            if (hasPrereqs) {
                tooltip.select('.math-prereq').text(finalPrereqs);
            }

            // 3. Trigger MathJax to typeset the content of the tooltip
            if (window.MathJax && window.MathJax.typesetPromise) {
                if (MathJax.typesetClear) { MathJax.typesetClear([tooltip.node()]); }
                MathJax.typesetPromise([tooltip.node()]).catch(err => {
                    console.error('MathJax typesetting failed:', err);
                });
            }
        }

        function hideTooltipIfNotPinned() {
            if (!pinned) {
                tooltip.style("display", "none");
            }
        }

        node
            .on("mouseover", (event, d) => {
                if (pinned) return;
                renderNodeTooltip(event, d);
            })
            .on("mouseout", () => {
                hideTooltipIfNotPinned();
            })
            .on("click", (event, d) => {
                event.stopPropagation();
                pinned = true;
                pinnedNode = d;
                renderNodeTooltip(event, d);
            });

        link.on("mouseover", (event, d) => {
            if (pinned) return;
            const dependencyType = (d.type || 'DEPENDS ON').replace(/_/g, ' ').toUpperCase();

            // 1. Clean and prepare content
            const rawDependency = d.dependency || 'N/A';
            const cleanedDependency = cleanLatexForDisplay(rawDependency);
            const finalDependency = wrapTeXOutsideMath(cleanedDependency);

            // 2. Set complete HTML with math content
            tooltip.style("display", "block")
                .html(`<h4>Dependency Link</h4>
                       <p>${d.source.display_name} <br>
                          <span class="edge-type">→ ${dependencyType} →</span> <br>
                          ${d.target.display_name}</p>
                       <p><strong>Justification:</strong></p>
                       <div class="math-content math-dep"></div>`)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 28) + "px");

            // Fill math text content
            tooltip.select('.math-dep').text(finalDependency);

            // 3. Trigger MathJax
            if (window.MathJax && window.MathJax.typesetPromise) {
                if (MathJax.typesetClear) { MathJax.typesetClear([tooltip.node()]); }
                MathJax.typesetPromise([tooltip.node()]).catch(err => {
                    console.error('MathJax typesetting failed:', err);
                });
            }
        }).on("mouseout", () => {
            hideTooltipIfNotPinned();
        });

        simulation.on("tick", () => {
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            label.attr("x", d => d.x).attr("y", d => d.y);
        });

        const legendContainer = d3.select("#legend-container");
        nodeTypes.forEach(type => {
            const item = legendContainer.append("div").attr("class", "legend-item");
            item.append("div").attr("class", "legend-color").style("background-color", nodeColors[type]);
            item.append("span").text(type.charAt(0).toUpperCase() + type.slice(1));
        });

        let isPlaying = true;
        d3.select("#play-pause").on("click", () => {
            if (isPlaying) {
                simulation.stop();
                d3.select("#play-pause").text("▶️ Play");
            } else {
                simulation.alpha(0.3).restart();
                d3.select("#play-pause").text("⏸️ Pause");
            }
            isPlaying = !isPlaying;
        });
        d3.select("#reset").on("click", () => {
            simulation.alpha(1).restart();
            if (!isPlaying) { d3.select("#play-pause").dispatch('click'); }
        });
        d3.select("#center").on("click", () => svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity));

        function dragstarted(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
        function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
        function dragended(event, d) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }
    </script>
</body>
</html>
"""
    # Safely extract data from the input dictionary
    stats = graph_data.get("stats", {})
    arxiv_id = graph_data.get("arxiv_id", "N/A")

    # Format graph data for JSON embedding
    graph_data_json = json.dumps(
        {"nodes": graph_data.get("nodes", []), "edges": graph_data.get("edges", [])},
        indent=2,
    )

    # Fill placeholders using a safe replace method
    html_content = (
        html_template.replace("__ARXIV_ID__", str(arxiv_id))
        .replace("__TIMESTAMP__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        .replace("__NODE_COUNT__", str(stats.get("node_count", 0)))
        .replace("__EDGE_COUNT__", str(stats.get("edge_count", 0)))
        .replace("__GRAPH_DATA_JSON__", graph_data_json)
    )

    try:
        output_path.write_text(html_content, encoding="utf-8")
        logger.success(f"Interactive visualization saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization HTML file: {e}")
