import * as d3 from 'd3';
import { typesetMath } from './mathjax';

function cleanLatex(content?: string | null) {
    if (!content) return '';

    // Backend strings often contain double-escaped backslashes (e.g. "\\\\alpha");
    // MathJax expects single backslashes ("\\alpha").
    const normalized = String(content).replace(/\\\\/g, '\\');

    return normalized.replace(/\\label\{[^}]*\}/g, '').trim();
}

export function setupLegends(
    nodeTypes: string[],
    edgeTypes: string[],
    nodeColors: Record<string, string>,
    edgeColors: Record<string, string>,
    state: any,
    actions: any,
) {
    const nodeLegendContainer = d3.select('#node-legend-container');
    nodeLegendContainer.selectAll('*').remove();

    nodeTypes.forEach((type) => {
        const item = nodeLegendContainer.append('div').attr('class', 'legend-item').attr('id', `legend-item-${type}`);
        item.append('div').attr('class', 'legend-color').style('background-color', nodeColors[type]);
        item.append('span').text(type.charAt(0).toUpperCase() + type.slice(1));

        item.on('click', () => {
            if (state.pinned) return;
            if (state.hiddenTypes.has(type)) {
                state.hiddenTypes.delete(type);
                item.classed('inactive', false);
            } else {
                state.hiddenTypes.add(type);
                item.classed('inactive', true);
            }
            actions.updateVisibility();
        });
    });

    const edgeLegendContainer = d3.select('#edge-legend-container');
    edgeLegendContainer.selectAll('*').remove();

    edgeTypes.forEach((type) => {
        const item = edgeLegendContainer.append('div').attr('class', 'legend-item');
        item.append('div').attr('class', 'edge-legend-line').style('background-color', edgeColors[type]);
        item.append('span').text(type.replace(/_/g, ' '));
    });
}

export async function renderNodeTooltip(tooltipEl: HTMLDivElement, event: any, d: any) {
    const finalPreview = cleanLatex(d.content_preview || d.content || 'N/A');

    tooltipEl.style.display = 'block';
    tooltipEl.innerHTML = `<h4>${d.display_name || d.label || d.id}</h4><div class="math-content">${finalPreview}</div>`;
    tooltipEl.style.left = `${event.offsetX + 15}px`;
    tooltipEl.style.top = `${event.offsetY + 15}px`;

    await typesetMath([tooltipEl]);
}

export function hideTooltip(tooltipEl: HTMLDivElement) {
    tooltipEl.style.display = 'none';
}

export async function updateInfoPanel(
    infoPanelEl: HTMLDivElement,
    infoTitleEl: HTMLDivElement,
    infoBodyEl: HTMLDivElement,
    d: any,
    state: any,
    actions: any,
) {
    infoTitleEl.textContent = d.display_name || d.label || d.id;

    const proofControls = state?.proofMode
        ? `
          <div class="proof-controls-inline">
            <button id="proof-unfold-less" class="depth-btn">← Unfold Less</button>
            <span style="color: var(--secondary-text); font-family: Inter, system-ui, sans-serif; font-size: 12px;">Depth: ${state.proofDepth}</span>
            <button id="proof-unfold-more" class="depth-btn">Unfold More →</button>
          </div>
        `
        : `
          <div class="proof-action">
            <button id="proof-explore" class="depth-btn depth-btn--primary">Explore Proof Path</button>
          </div>
        `;

    let infoHTML = `${proofControls}<h4>Preview</h4><div class="math-content">${cleanLatex(d.content_preview || d.content || 'N/A')}</div>`;

    if (d.prerequisites_preview) {
        infoHTML += `<h4>Prerequisites</h4><div class="math-content">${cleanLatex(d.prerequisites_preview)}</div>`;
    }

    if (d.position && typeof d.position.line_start !== 'undefined') {
        infoHTML += `<div style="margin-top:10px; font-size:12px; color: var(--secondary-text)">Source position: line ${d.position.line_start}${d.position.line_end ? `–${d.position.line_end}` : ''}</div>`;
    }

    infoBodyEl.innerHTML = infoHTML;
    infoPanelEl.classList.add('visible');

    // Wire handlers
    const exploreBtn = document.getElementById('proof-explore');
    if (exploreBtn) {
        exploreBtn.onclick = () => actions.enterProofMode(d.id);
    }

    const lessBtn = document.getElementById('proof-unfold-less');
    if (lessBtn) {
        lessBtn.onclick = () => actions.unfoldLess();
    }

    const moreBtn = document.getElementById('proof-unfold-more');
    if (moreBtn) {
        moreBtn.onclick = () => actions.unfoldMore();
    }

    await typesetMath([infoBodyEl]);
}

export function hideInfoPanel(infoPanelEl: HTMLDivElement) {
    infoPanelEl.classList.remove('visible');
}
