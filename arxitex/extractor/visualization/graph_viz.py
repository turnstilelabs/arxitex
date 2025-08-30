import json
from pathlib import Path
from typing import Dict
from loguru import logger
from datetime import datetime

def create_visualization_html(graph_data: Dict, output_path: Path) -> None:
    """
    Creates an HTML file with an interactive D3.js graph visualization.

    This version is enhanced to:
    - Use MathJax to render LaTeX formulas in tooltips.
    - Correctly display the specific dependency type on edges.
    - Display both the content preview and prerequisite definitions for nodes.
    - Include improved styling for better readability.

    Args:
        graph_data: A dictionary containing the graph data (nodes, edges) and stats.
        output_path: The path where the HTML file will be saved.
    """
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv Paper Dependency Graph - {arxiv_id}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    
    <!-- MATHJAX INTEGRATION: Add MathJax scripts to the head -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; color: #333; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        h1 {{ margin: 0 0 10px 0; }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px; }}
        .stat {{ background: #e9ecef; padding: 8px 12px; border-radius: 4px; font-size: 14px; }}
        .graph-container {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
        .controls {{ padding: 15px; background: #f8f9fa; border-bottom: 1px solid #dee2e6; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
        .controls button {{ padding: 6px 12px; border: 1px solid #dee2e6; background: white; border-radius: 4px; cursor: pointer; font-size: 14px; transition: background-color 0.2s; }}
        .controls button:hover {{ background: #e9ecef; }}
        .legend {{ display: flex; gap: 15px; align-items: center; flex-wrap: wrap; margin-left: auto; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; }}
        #graph {{ width: 100%; height: 70vh; cursor: grab; }}
        #graph:active {{ cursor: grabbing; }}
        .node {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .node:hover {{ stroke: #000; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; stroke-width: 1.5px; transition: stroke-width 0.2s, stroke-opacity 0.2s; cursor: pointer; marker-end: url(#arrowhead); }}
        .link:hover {{ stroke-width: 4px; stroke-opacity: 1; }}
        .node-label {{ font-size: 12px; fill: #333; text-anchor: middle; pointer-events: none; font-weight: 500; }}
        .tooltip {{ position: absolute; background: rgba(0, 0, 0, 0.9); color: white; padding: 12px; border-radius: 6px; font-size: 14px; pointer-events: none; max-width: 450px; z-index: 1000; line-height: 1.6; border: 1px solid #333; }}
        .tooltip h4 {{ margin: 0 0 10px 0; font-size: 16px; border-bottom: 1px solid #555; padding-bottom: 8px; }}
        .tooltip p {{ margin: 5px 0; }}
        .tooltip .id-label {{ color: #aaa; font-family: monospace; font-size: 11px; }}
        .tooltip .edge-type {{ font-weight: bold; color: #ffc107; text-transform: uppercase; }}
        .tooltip b {{ color: #a2d2ff; }} /* Make terms in prerequisites stand out */
    </style>
</head>
<body>
    <div class="header">
        <h1>arXiv Paper Dependency Graph</h1>
        <p><strong>Paper ID:</strong> {arxiv_id} | <strong>Generated:</strong> {timestamp}</p>
        <div class="stats">
            <div class="stat"><strong>{node_count}</strong> artifacts</div>
            <div class="stat"><strong>{edge_count}</strong> references</div>
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
        const graphData = {graph_data_json};
        const nodeTypes = [...new Set(graphData.nodes.map(d => d.type))];
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        const nodeColors = nodeTypes.reduce((acc, type) => {{
            acc[type] = colorScale(type);
            return acc;
        }}, {{}});

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

        // --- UPDATED TOOLTIP LOGIC FOR NODES ---
        node.on("mouseover", (event, d) => {{
            // Build the prerequisites section only if it exists
            let prereqHtml = d.prerequisites_preview ? 
                `<h4>Prerequisites</h4><p>${{d.prerequisites_preview}}</p>` : '';

            tooltip.style("display", "block")
                .html(`<h4>${{d.display_name}}</h4>
                       <p><span class="id-label">ID: ${{d.id}} | Label: ${{d.label || 'N/A'}}</span></p>
                       <p><strong>Preview:</strong> ${{d.content_preview}}</p>
                       ${{prereqHtml}}`)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 28) + "px");

            // MATHJAX INTEGRATION: Tell MathJax to typeset the content of the tooltip.
            if (window.MathJax) {{
                MathJax.typesetPromise([tooltip.node()]);
            }}
        }}).on("mouseout", () => {{
            tooltip.style("display", "none");
        }});

        // --- UPDATED TOOLTIP LOGIC FOR LINKS ---
        link.on("mouseover", (event, d) => {{
            // Use the reliable 'type' field from Edge.to_dict() and format it.
            const dependencyType = (d.type || 'DEPENDS ON').replace('_', ' ').toUpperCase();

            tooltip.style("display", "block")
                .html(`<h4>Dependency Link</h4>
                       <p>${{d.source.display_name}} <br>
                          <span class="edge-type">→ ${{dependencyType}} →</span> <br>
                          ${{d.target.display_name}}</p>
                       <p><strong>Justification:</strong> ${{d.dependency || 'N/A'}}</p>`)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 28) + "px");
            
            // Also typeset math in the justification text, if any.
            if (window.MathJax) {{
                MathJax.typesetPromise([tooltip.node()]);
            }}
        }}).on("mouseout", () => {{
            tooltip.style("display", "none");
        }});

        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            label.attr("x", d => d.x).attr("y", d => d.y);
        }});
        
        const legendContainer = d3.select("#legend-container");
        nodeTypes.forEach(type => {{
            const item = legendContainer.append("div").attr("class", "legend-item");
            item.append("div").attr("class", "legend-color").style("background-color", nodeColors[type]);
            item.append("span").text(type.charAt(0).toUpperCase() + type.slice(1));
        }});

        let isPlaying = true;
        d3.select("#play-pause").on("click", () => {{
            if (isPlaying) {{
                simulation.stop();
                d3.select("#play-pause").text("▶️ Play");
            }} else {{
                simulation.alpha(0.3).restart();
                d3.select("#play-pause").text("⏸️ Pause");
            }}
            isPlaying = !isPlaying;
        }});
        d3.select("#reset").on("click", () => {{
            simulation.alpha(1).restart();
            if (!isPlaying) {{ d3.select("#play-pause").dispatch('click'); }}
        }});
        d3.select("#center").on("click", () => svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity));
        
        function dragstarted(event, d) {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
        function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
        function dragended(event, d) {{ if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}
    </script>
</body>
</html>
"""
    
    # Safely extract data from the input dictionary
    nodes_for_json = graph_data.get('nodes', [])
    edges_for_json = graph_data.get('edges', [])
    stats = graph_data.get('stats', {})
    arxiv_id = graph_data.get('arxiv_id', 'N/A')

    # Format graph data for JSON embedding
    graph_data_json = json.dumps(
        {"nodes": nodes_for_json, "edges": edges_for_json},
        indent=2
    )
    
    # Format the HTML template with all the necessary data
    html_content = html_template.format(
        arxiv_id=arxiv_id,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        node_count=stats.get('node_count', 0),
        edge_count=stats.get('edge_count', 0),
        graph_data_json=graph_data_json
    )

    try:
        output_path.write_text(html_content, encoding='utf-8')
        logger.success(f"Interactive visualization saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization HTML file: {e}")