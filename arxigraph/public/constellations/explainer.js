/* eslint-disable @typescript-eslint/no-unused-vars */
// Copied from ../constellations/assets/explainer.js (lightly adjusted path-wise via public hosting)
(function () {
    try {
        console.log('[Explainer] script loaded');
    } catch (_) { }
    var model = null;
    try {
        var dataEl = document.getElementById('distill-data');
        if (dataEl) model = JSON.parse(dataEl.textContent || '{}');
    } catch (e) {
        console.error('[Explainer] Failed to parse model JSON:', e);
    }

    var menu = document.getElementById('explainer-menu');
    var setKeyBtn = document.getElementById('set-api-key');

    try {
        if (menu && menu.parentElement !== document.body) document.body.appendChild(menu);
        if (menu && window.getComputedStyle(menu).position !== 'fixed') menu.style.position = 'fixed';
    } catch (e) { }

    var highlightLayer = document.getElementById('highlight-layer');
    var pinLayer = document.getElementById('pin-layer');
    var commentPanel = document.getElementById('comment-panel');
    var threadsContainer = document.getElementById('threads');
    var closePanelBtn = document.getElementById('close-panel');

    try {
        if (highlightLayer && highlightLayer.parentElement !== document.body) document.body.appendChild(highlightLayer);
        if (pinLayer && pinLayer.parentElement !== document.body) document.body.appendChild(pinLayer);
        if (highlightLayer && window.getComputedStyle(highlightLayer).position !== 'fixed') highlightLayer.style.position = 'fixed';
        if (pinLayer && window.getComputedStyle(pinLayer).position !== 'fixed') pinLayer.style.position = 'fixed';
    } catch (_) { }

    var threads = new Map();
    var activeThreadId = null;
    var threadSeq = 1;
    var lastActionAt = 0;
    var panelOpenedAt = 0;

    function openPanel() {
        if (commentPanel) {
            commentPanel.classList.add('open');
            panelOpenedAt = Date.now();
            try {
                recomputeAllPositions();
            } catch (_) { }
        }
    }
    function closePanel() {
        if (commentPanel) {
            commentPanel.classList.remove('open');
            panelOpenedAt = 0;
            try {
                recomputeAllPositions();
            } catch (_) { }
        }
    }

    if (closePanelBtn) closePanelBtn.addEventListener('click', function () {
        closePanel();
    });
    if (threadsContainer) {
        threadsContainer.addEventListener('click', function (e) {
            var card = e.target && e.target.closest && e.target.closest('.thread-card');
            if (!card) return;
            var id = card.getAttribute('data-id');
            if (!id) return;
            selectThread(id);
        });
    }

    function selectThread(id) {
        activeThreadId = id;
        if (!threadsContainer) return;
        var cards = threadsContainer.querySelectorAll('.thread-card');
        for (var i = 0; i < cards.length; i++) cards[i].classList.remove('active');
        var t = threads.get(id);
        if (t && t.cardEl) t.cardEl.classList.add('active');
    }

    function scrollThreadIntoView(id) {
        var t = threads.get(id);
        if (t && t.cardEl && typeof t.cardEl.scrollIntoView === 'function') {
            t.cardEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    function rectsFromRange(range) {
        try {
            var list = range.getClientRects();
            var arr = [];
            for (var i = 0; i < list.length; i++) {
                var r = list[i];
                arr.push({ left: r.left, top: r.top, width: r.width, height: r.height, right: r.right, bottom: r.bottom });
            }
            return arr;
        } catch (_) {
            return [];
        }
    }

    function clearThreadVisuals(t) {
        if (!t) return;
        if (t.pinEl) t.pinEl.remove();
        if (t.highlightEls) {
            for (var i = 0; i < t.highlightEls.length; i++) t.highlightEls[i].remove();
        }
        t.pinEl = null;
        t.highlightEls = [];
    }

    function createThreadVisuals(id) {
        var t = threads.get(id);
        if (!t) return;
        clearThreadVisuals(t);
        var rects = rectsFromRange(t.range);
        t.highlightEls = [];
        for (var i = 0; i < rects.length; i++) {
            var hr = document.createElement('div');
            hr.className = 'highlight-rect';
            hr.style.left = rects[i].left + 'px';
            hr.style.top = rects[i].top + 'px';
            hr.style.width = rects[i].width + 'px';
            hr.style.height = rects[i].height + 'px';
            if (highlightLayer) highlightLayer.appendChild(hr);
            t.highlightEls.push(hr);
        }
        if (rects.length) {
            var pin = document.createElement('button');
            pin.className = 'pin';
            pin.setAttribute('type', 'button');
            var endRect = rects[rects.length - 1];
            var anc = t.range.commonAncestorContainer;
            var ancEl = anc && anc.nodeType === 1 ? anc : anc ? anc.parentElement : null;
            var blockEl = typeof findBlockElement === 'function' ? findBlockElement(ancEl) : null;
            var blockRect = blockEl && blockEl.getBoundingClientRect ? blockEl.getBoundingClientRect() : null;
            var pinX = blockRect ? blockRect.right - 18 - 20 : endRect.right + 20;
            pin.style.left = pinX + 'px';
            pin.style.top = endRect.top + Math.max(0, (endRect.height - 18) / 2) + 'px';
            pin.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                openPanel();
                selectThread(id);
                scrollThreadIntoView(id);
            });
            if (pinLayer) pinLayer.appendChild(pin);
            t.pinEl = pin;
        }
    }

    function recomputeAllPositions() {
        threads.forEach(function (t) {
            createThreadVisuals(t.id);
        });
    }

    var moTimer = null;
    try {
        var root = document.querySelector('.doc') || document.body;
        var mo = new MutationObserver(function () {
            if (moTimer) cancelAnimationFrame(moTimer);
            moTimer = requestAnimationFrame(recomputeAllPositions);
        });
        mo.observe(root, { childList: true, subtree: true, characterData: true });
    } catch (_) { }

    window.addEventListener('scroll', recomputeAllPositions, true);
    window.addEventListener('resize', recomputeAllPositions);

    function getApiKey() {
        try {
            return localStorage.getItem('openai_api_key') || '';
        } catch (_) {
            return '';
        }
    }
    function setApiKey(k) {
        try {
            if (!k) localStorage.removeItem('openai_api_key');
            else localStorage.setItem('openai_api_key', String(k).trim());
        } catch (_) { }
    }

    window.showKeyModal = function () {
        try {
            var modal = document.getElementById('key-modal');
            var input = document.getElementById('key-input');
            if (!modal || !input) return;
            input.value = getApiKey();
            modal.style.display = 'flex';
            setTimeout(function () {
                try {
                    input.focus();
                } catch (_) { }
            }, 0);
        } catch (e) {
            console.error(e);
        }
    };
    window.hideKeyModal = function () {
        var modal = document.getElementById('key-modal');
        if (modal) modal.style.display = 'none';
    };
    function keySave() {
        var input = document.getElementById('key-input');
        var v = input && input.value ? input.value.trim() : '';
        setApiKey(v);
        var btn = document.getElementById('set-api-key');
        if (btn) btn.title = v ? 'API key saved' : 'Set OpenAI API key';
        window.hideKeyModal();
    }
    (function wireKeyModal() {
        var elSave = document.getElementById('key-save');
        if (elSave) elSave.addEventListener('click', keySave);
        var elCancel = document.getElementById('key-cancel');
        if (elCancel) elCancel.addEventListener('click', window.hideKeyModal);
        var elInput = document.getElementById('key-input');
        if (elInput)
            elInput.addEventListener('keydown', function (e) {
                if (e.key === 'Enter') keySave();
                if (e.key === 'Escape') window.hideKeyModal();
            });
        if (setKeyBtn) setKeyBtn.addEventListener('click', window.showKeyModal);
    })();

    function hideMenu() {
        if (menu) {
            menu.style.display = 'none';
            menu.innerHTML = '';
        }
    }

    function buildMenu(x, y, onPick) {
        try {
            console.log('[Explainer] buildMenu at', x, y);
        } catch (_) { }
        if (!menu) return;
        var items = [
            { mode: 'simplify', label: 'Explain this in simpler terms' },
            { mode: 'intuition', label: 'What is the key intuition here?' },
            { mode: 'expand', label: 'Expand this step' },
        ];
        var html = '';
        for (var i = 0; i < items.length; i++) html += '<button data-mode="' + items[i].mode + '">' + items[i].label + '</button>';
        menu.innerHTML = html;
        var left = Math.max(8, Math.min(window.innerWidth - 260, x));
        menu.style.display = 'block';
        var mh = menu.offsetHeight || 140;
        var top = Math.max(8, Math.min(window.innerHeight - mh - 8, y));
        menu.style.left = left + 'px';
        menu.style.top = top + 'px';
        var buttons = menu.querySelectorAll('button');
        for (var j = 0; j < buttons.length; j++) {
            buttons[j].addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                var m = this.getAttribute('data-mode');
                hideMenu();
                try {
                    openPanel();
                } catch (_) { }
                onPick(m);
            });
        }
    }

    function getSelectionInfo() {
        var sel = window.getSelection ? window.getSelection() : null;
        if (!sel || sel.isCollapsed) return null;

        var text = String(sel.toString() || '').trim();
        var range = sel.rangeCount ? sel.getRangeAt(0) : null;

        function isInMath(node) {
            try {
                var el = node && (node.nodeType === 1 ? node : node.parentElement);
                while (el && el !== document.body) {
                    if (el.tagName && (el.tagName.toLowerCase() === 'mjx-container' || el.tagName.toLowerCase() === 'mjx-math')) return true;
                    el = el.parentElement;
                }
            } catch (_) { }
            return false;
        }
        var inMath = !!(range && isInMath(range.commonAncestorContainer));

        if ((!text || (inMath ? text.length < 1 : text.length < 2)) && range) {
            try {
                var frag = range.cloneContents();
                var t2 = (frag && frag.textContent ? String(frag.textContent) : '').trim();
                if (t2 && (inMath ? t2.length >= 1 : t2.length >= 2)) text = t2;
            } catch (e) { }
        }

        if ((!text || (inMath ? text.length < 1 : text.length < 2)) && range && range.commonAncestorContainer) {
            var ctxNode = range.commonAncestorContainer.nodeType === 1 ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
            var block = findBlockElement(ctxNode);
            var ctxText = block && typeof block.innerText === 'string' ? block.innerText.trim() : '';
            if (ctxText && (inMath ? ctxText.length >= 1 : ctxText.length >= 2)) text = ctxText.slice(0, inMath ? 120 : 200);
        }

        if (!range) return null;
        if (inMath ? text.length < 1 : text.length < 2) return null;

        try {
            var ancNode = range.commonAncestorContainer.nodeType === 1 ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
            if ((commentPanel && commentPanel.contains(ancNode)) || (threadsContainer && threadsContainer.contains(ancNode))) {
                return null;
            }
        } catch (_) { }

        var rect = range.getBoundingClientRect();
        if ((!rect || (rect.top === 0 && rect.left === 0 && rect.width === 0 && rect.height === 0)) && typeof range.getClientRects === 'function') {
            var list = range.getClientRects();
            if (list && list.length) rect = list[0];
        }
        return { sel: sel, range: range, rect: rect, text: text };
    }

    function findBlockElement(node) {
        function isBlock(el) {
            if (!el) return false;
            if (el.classList && (el.classList.contains('math-content') || el.classList.contains('result-item') || el.classList.contains('def-item') || el.classList.contains('target-item'))) return true;
            var tag = el.tagName || '';
            return ['P', 'DIV', 'SECTION', 'ARTICLE', 'H1', 'H2', 'H3', 'LI', 'OL', 'UL'].indexOf(tag) !== -1;
        }
        var el = node;
        while (el && el !== document.body && !isBlock(el)) el = el.parentElement;
        return el || document.body;
    }

    function nearestInsertionPoint(range) {
        var container = range.commonAncestorContainer && range.commonAncestorContainer.nodeType === 1 ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
        return findBlockElement(container);
    }

    function insertExplainerCard(blockEl, title) {
        var card = document.createElement('div');
        card.className = 'explainer-card';
        card.innerHTML =
            '<header>' +
            '<div class="title">AI explanation — ' +
            title +
            '</div>' +
            '<div class="actions">' +
            '<button data-act="copy">Copy</button>' +
            '<button data-act="regen">Regenerate</button>' +
            '<button data-act="remove">Remove</button>' +
            '</div>' +
            '</header>' +
            '<div class="content math-content"><em>Working…</em></div>';
        blockEl.insertAdjacentElement('afterend', card);
        return card;
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>]/g, function (c) {
            return { '&': '&', '<': '<', '>': '>' }[c];
        });
    }

    function typesetMath(el) {
        try {
            if (window.MathJax && window.MathJax.typesetPromise) return window.MathJax.typesetPromise([el]);
        } catch (e) { }
        return Promise.resolve();
    }

    function openaiExplain(payload) {
        return new Promise(function (resolve, reject) {
            var apiKey = getApiKey();
            if (!apiKey) {
                reject(new Error('Missing API key. Click the gear button to set it.'));
                return;
            }
            var sys =
                'You are The Explainer for mathematical content. Be precise and correct; preserve meaning; include small LaTeX where helpful; avoid fabrications; if unsure, say so.';
            var parts = [];
            parts.push('Mode: ' + payload.mode);
            parts.push('Selection: ' + payload.selection);
            if (payload.localContext) parts.push('Local context: ' + payload.localContext);
            if (payload.target) parts.push('Target: ' + (payload.target.title || '') + ': ' + (payload.target.content || ''));
            if (payload.definitions && payload.definitions.length) parts.push('Definitions (subset):\n- ' + payload.definitions.slice(0, 6).join('\n- '));
            if (payload.supporting && payload.supporting.length)
                parts.push(
                    'Supporting (subset):\n- ' +
                    payload.supporting
                        .slice(0, 3)
                        .map(function (r) {
                            return (r.title || '') + ': ' + (r.content || '');
                        })
                        .join('\n- '),
                );
            var prompt = parts.join('\n\n');

            fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + apiKey },
                body: JSON.stringify({
                    model: 'gpt-4o-mini',
                    temperature: 0.2,
                    messages: [
                        { role: 'system', content: sys },
                        { role: 'user', content: prompt },
                    ],
                }),
            })
                .then(function (resp) {
                    if (!resp.ok)
                        return resp.json().then(function (err) {
                            throw new Error('HTTP ' + resp.status + ' ' + ((err && err.error && err.error.message) || resp.statusText));
                        });
                    return resp.json();
                })
                .then(function (data) {
                    var text = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content || '';
                    resolve(String(text).trim());
                })
                .catch(reject);
        });
    }

    function handleAction(mode, selInfo) {
        try {
            hideMenu();
        } catch (_) { }
        lastActionAt = Date.now();
        var title = mode === 'simplify' ? 'Simpler terms' : mode === 'intuition' ? 'Key intuition' : 'Expanded step';

        var ctxEl = findBlockElement(selInfo.range.commonAncestorContainer);
        var ctxText = ctxEl && typeof ctxEl.innerText === 'string' ? ctxEl.innerText : '';
        var localContext = ctxText ? ctxText.trim().slice(0, 4000) : '';

        var defs = model && model.definitions ? model.definitions.slice(0, 6) : [];
        var supp = model && model.supporting ? model.supporting.slice(0, 3) : [];
        var payload = {
            mode: mode,
            selection: selInfo.text.slice(0, 4000),
            localContext: localContext,
            target: model && model.target,
            definitions: defs,
            supporting: supp,
        };

        var id = 't' + threadSeq++;
        var t = {
            id: id,
            range: selInfo.range.cloneRange ? selInfo.range.cloneRange() : selInfo.range,
            pinEl: null,
            highlightEls: [],
            cardEl: null,
        };
        threads.set(id, t);

        if (threadsContainer) {
            var card = document.createElement('div');
            card.className = 'thread-card active';
            card.setAttribute('data-id', id);
            card.innerHTML =
                '<header>' +
                '<div class="title">AI explanation — ' +
                title +
                '</div>' +
                '<div class="actions">' +
                '<button class="icon-btn" data-act="remove" title="Remove">×</button>' +
                '</div>' +
                '</header>' +
                '<div class="content math-content"><em>Working…</em></div>';
            threadsContainer.appendChild(card);
            t.cardEl = card;
        }

        openPanel();
        selectThread(id);
        scrollThreadIntoView(id);
        createThreadVisuals(id);

        var contentEl = t.cardEl ? t.cardEl.querySelector('.content') : null;
        function doExplain() {
            if (contentEl) contentEl.innerHTML = '<em>Working…</em>';
            openaiExplain(payload)
                .then(function (reply) {
                    if (contentEl) contentEl.innerHTML = '<div>' + reply + '</div>';
                    return typesetMath(contentEl);
                })
                .then(function () {
                    try {
                        recomputeAllPositions();
                    } catch (_) { }
                })
                .catch(function (e) {
                    if (contentEl) contentEl.innerHTML = '<span style="color:#c00">' + escapeHtml(e.message || String(e)) + '</span>';
                });
        }
        doExplain();

        if (t.cardEl) {
            var actions = t.cardEl.querySelector('.actions');
            if (actions)
                actions.addEventListener('click', function (evt) {
                    var act = evt.target.getAttribute('data-act');
                    if (act === 'remove') {
                        clearThreadVisuals(t);
                        if (t.cardEl) t.cardEl.remove();
                        threads.delete(id);
                        if (activeThreadId === id) {
                            activeThreadId = null;
                            var last = null;
                            threads.forEach(function (v) {
                                last = v;
                            });
                            if (last) {
                                selectThread(last.id);
                            }
                        }
                    }
                });
        }
    }

    function onSelectionEvent() {
        if (Date.now() - lastActionAt < 600) {
            hideMenu();
            return;
        }
        var info = getSelectionInfo();
        if (!info) {
            hideMenu();
            return;
        }
        var x = info.rect.left + info.rect.width / 2;
        var y = Math.max(8, info.rect.top - 10);
        buildMenu(x, y, function (mode) {
            handleAction(mode, info);
        });
    }

    document.addEventListener('mouseup', function () {
        setTimeout(onSelectionEvent, 30);
    });

    document.addEventListener('scroll', hideMenu, true);
    document.addEventListener('click', function (e) {
        if (menu && !menu.contains(e.target)) hideMenu();
        if (commentPanel && commentPanel.classList.contains('open')) {
            if (Date.now() - panelOpenedAt < 300) return;
            var isPin = e.target && e.target.closest && e.target.closest('.pin');
            if (!isPin && !commentPanel.contains(e.target)) closePanel();
        }
    });

    window.addEventListener(
        'contextmenu',
        function (e) {
            var info = getSelectionInfo();
            if (info) {
                e.preventDefault();
                var y = Math.max(8, info.rect.top - 10);
                buildMenu(e.clientX, y, function (mode) {
                    handleAction(mode, info);
                });
            }
        },
        { capture: true },
    );

    window.addEventListener('keydown', function (e) {
        var isShortcut = (e.key === 'e' || e.key === 'E') && (e.ctrlKey || e.metaKey);
        if (!isShortcut) return;
        var info = getSelectionInfo();
        if (!info) return;
        e.preventDefault();
        buildMenu(info.rect.left + info.rect.width / 2, Math.max(8, info.rect.top - 10), function (mode) {
            handleAction(mode, info);
        });
    });
})();
