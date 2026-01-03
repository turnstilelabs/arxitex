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
    // Human-friendly label: replace underscores with spaces, capitalize first letter.
    const prettyLabelBase = type.replace(/_/g, ' ');
    const prettyLabel = prettyLabelBase.charAt(0).toUpperCase() + prettyLabelBase.slice(1);
    item.append('span').text(prettyLabel);

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

  // Edge types removed from the legend UI.
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
  const nodeTitle = d.display_name || d.label || d.id;
  infoTitleEl.innerHTML = `
      <span class="info-title-text">${nodeTitle}</span>
      <button
        id="report-node-issue-header"
        class="info-title-flag-btn"
        title="Suggest a correction for this node"
        aria-label="Suggest a correction for this node"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M4 22V4" />
          <path d="M4 4h12l-1.5 4L20 12H4" />
        </svg>
      </button>
    `;

  const proofControls = state?.proofMode
    ? `
          <div class="proof-controls-inline">
            <button id="proof-distill" class="depth-btn depth-btn--primary proof-btn-center">Generate Distilled Proof</button>

            <div class="proof-depth-controls" aria-label="Unfolding depth controls">
              <button
                id="proof-unfold-less"
                class="depth-btn depth-btn--icon"
                title="Unfold less (decrease depth)"
                aria-label="Unfold less (decrease depth)"
              >
                −
              </button>
              <div class="proof-depth-label" title="Current unfolding depth" aria-label="Current unfolding depth">
                Depth: ${Number(state?.proofDepth ?? 0)}
              </div>
              <button
                id="proof-unfold-more"
                class="depth-btn depth-btn--icon"
                title="Unfold more (increase depth)"
                aria-label="Unfold more (increase depth)"
              >
                +
              </button>
            </div>
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

  // Source position removed from UI.

  infoBodyEl.innerHTML = infoHTML;
  infoPanelEl.classList.add('visible');

  // Wire handlers
  const exploreBtn = document.getElementById('proof-explore');
  if (exploreBtn) {
    exploreBtn.onclick = () => actions.enterProofMode(d.id);
  }



  const reportHeaderBtn = document.getElementById('report-node-issue-header');
  if (reportHeaderBtn) {
    reportHeaderBtn.onclick = () => actions.reportNodeIssue(d);
  }

  const lessBtn = document.getElementById('proof-unfold-less');
  if (lessBtn) {
    lessBtn.onclick = () => actions.unfoldLess();
  }

  const moreBtn = document.getElementById('proof-unfold-more');
  if (moreBtn) {
    moreBtn.onclick = () => actions.unfoldMore();
  }

  const distillBtn = document.getElementById('proof-distill');
  if (distillBtn) {
    distillBtn.onclick = () => actions.generateDistilledProof();
  }

  await typesetMath([infoBodyEl]);
}

export function hideInfoPanel(infoPanelEl: HTMLDivElement) {
  infoPanelEl.classList.remove('visible');
}
