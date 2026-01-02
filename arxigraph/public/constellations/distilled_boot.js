/* eslint-disable @typescript-eslint/no-unused-vars */
// Copied from ../constellations/assets/distilled_boot.js
(function () {
    var model = {};
    try {
        var dataEl = document.getElementById('distill-data');
        if (dataEl) model = JSON.parse(dataEl.textContent || '{}');
    } catch (e) {
        console.error('[DistilledBoot] Failed to parse model JSON:', e);
    }

    function sanitizeFilename(s) {
        return String(s || 'distilled-proof')
            .replace(/[^a-z0-9\-_]+/gi, '_')
            .slice(0, 120);
    }
    function downloadBlob(content, type, filename) {
        var blob = new Blob([content], { type: type });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function () {
            URL.revokeObjectURL(a.href);
            a.remove();
        }, 0);
    }
    function buildLatexFromModel(model) {
        var esc = function (s) {
            return s == null ? '' : String(s);
        };
        var lines = [];
        lines.push('\\documentclass[11pt]{article}');
        lines.push('\\usepackage{amsmath,amssymb,amsthm}');
        lines.push('\\usepackage[margin=1in]{geometry}');
        lines.push('\\title{Distilled Proof for: ' + esc(model.title).replace(/[\\{}]/g, ' ') + '}');
        lines.push('\\begin{document}');
        lines.push('\\maketitle');
        if (model.definitions && model.definitions.length) {
            lines.push('\\section*{Definitions and Notations}');
            model.definitions.forEach(function (d) {
                lines.push(esc(d));
                lines.push('');
            });
        }
        if (model.supporting && model.supporting.length) {
            lines.push('\\section*{Supporting Results}');
            model.supporting.forEach(function (r) {
                lines.push('\\subsection*{' + esc(r.title).replace(/[\\{}]/g, ' ') + '}');
                lines.push(esc(r.content));
                lines.push('');
            });
        }
        lines.push('\\section*{Target Theorem and Proof}');
        lines.push('\\subsection*{' + esc(model.target && model.target.title || '').replace(/[\\{}]/g, ' ') + '}');
        lines.push(esc(model.target && model.target.content || ''));
        lines.push('\\end{document}');
        return lines.join('\n');
    }

    (function setDocUrl() {
        try {
            var openerHref = (window.opener && window.opener.location && window.opener.location.href) || window.location.href;
            var url = new URL(openerHref);
            url.searchParams.set('distilled_doc', '1');
            if (model && model.target && model.target.id) url.searchParams.set('target', String(model.target.id));
            if (typeof model.depth !== 'undefined') url.searchParams.set('depth', String(model.depth));
            var __tgt = model && model.target ? model.target.id : null;
            var __dep = model ? model.depth : null;
            history.replaceState({ distilled_doc: true, target: __tgt, depth: __dep }, '', url);
        } catch (e) { }
    })();

    var texBtn = document.getElementById('download-tex');
    if (texBtn) {
        texBtn.addEventListener('click', function () {
            var tex = buildLatexFromModel(model);
            var base = 'Distilled-' + sanitizeFilename(model.title || (model.target && model.target.title) || 'proof');
            downloadBlob(tex, 'text/x-tex', base + '.tex');
        });
    }

    function getChildren(id) {
        var adj = (model && model.adj) || {};
        return adj[id] || [];
    }
    function getNodeInfo(id) {
        var gn = (model && model.graphNodes) || {};
        return gn[id] || null;
    }
    function ensureChildRendered(parentId, childId) {
        var container = document.querySelector('.child-results[data-parent="' + parentId + '"]');
        if (!container) return null;
        var exists = container.querySelector('.result-item[data-id="' + childId + '"]');
        if (exists) return exists;
        var info = getNodeInfo(childId);
        if (!info) return null;
        var div = document.createElement('div');
        div.className = 'result-item';
        div.setAttribute('data-id', childId);
        div.innerHTML =
            '<h3 style="font-size:15px">' +
            (info.title || childId) +
            ' ' +
            '<span class="artifact-controls" data-id="' +
            childId +
            '">' +
            '<button class="fold-btn" data-act="less" title="Unfold less">−</button>' +
            '<button class="fold-btn" data-act="more" title="Unfold more">+</button>' +
            '</span>' +
            '</h3>' +
            '<div class="math-content">' +
            (info.content || '<em>No statement available.</em>') +
            '</div>' +
            '<div class="child-results" data-parent="' +
            childId +
            '"></div>';
        container.appendChild(div);
        try {
            if (window.MathJax && window.MathJax.typesetPromise) MathJax.typesetPromise([div]);
        } catch (_) { }
        return div;
    }
    function expandFrom(id) {
        var item = document.querySelector('.result-item[data-id="' + id + '"]');
        if (!item) return;
        item.classList.remove('collapsed');
        var container = document.querySelector('.child-results[data-parent="' + id + '"]');
        if (!container) return;

        var existing = container.querySelectorAll('.result-item[data-id]');
        if (!existing || existing.length === 0) {
            var kids = getChildren(id) || [];
            for (var i = 0; i < kids.length; i++) ensureChildRendered(id, kids[i]);
            try {
                if (window.MathJax && window.MathJax.typesetPromise) MathJax.typesetPromise([container]);
            } catch (_) { }
            return;
        }

        for (var j = 0; j < existing.length; j++) {
            var childEl = existing[j];
            var cid = childEl.getAttribute('data-id');
            var gkids = getChildren(cid) || [];
            for (var k = 0; k < gkids.length; k++) ensureChildRendered(cid, gkids[k]);
        }
        try {
            if (window.MathJax && window.MathJax.typesetPromise) MathJax.typesetPromise([container]);
        } catch (_) { }
    }

    document.addEventListener(
        'click',
        function (e) {
            var btn = e.target && e.target.closest && e.target.closest('.fold-btn');
            if (!btn) return;
            var wrap = btn.closest('.artifact-controls');
            var id = wrap && wrap.getAttribute('data-id');
            if (!id) return;
            e.preventDefault();
            var act = btn.getAttribute('data-act');
            var item = document.querySelector('.result-item[data-id="' + id + '"]');
            if (!item) return;
            if (act === 'more') {
                expandFrom(id);
            } else {
                item.classList.add('collapsed');
            }
        },
        true,
    );
})();
