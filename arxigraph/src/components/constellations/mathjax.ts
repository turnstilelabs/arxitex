declare global {
    interface Window {
        MathJax?: any;
    }
}

export async function typesetMath(elements: Array<Element | null | undefined>) {
    try {
        if (typeof window === 'undefined') return;

        const filtered = elements.filter(Boolean) as Element[];
        if (!filtered.length) return;

        // Wait for MathJax to be ready (v3 exposes startup.promise)
        for (let i = 0; i < 20; i++) {
            const mj = window.MathJax;
            if (mj && typeof mj.typesetPromise === 'function') {
                if (mj.startup?.promise) {
                    await mj.startup.promise;
                }
                await mj.typesetPromise(filtered);
                return;
            }
            await new Promise((r) => setTimeout(r, 50));
        }
    } catch (e) {
        // Best effort only
        console.warn('MathJax typesetting failed', e);
    }
}
