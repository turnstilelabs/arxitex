/* eslint-disable @next/next/no-page-custom-font */

import type { Metadata } from 'next';
import Script from 'next/script';

import './globals.css';
// Constellations global styles (kept in components so graph UI can share it).
// Importing here avoids CSS `@import` path resolution issues in Turbopack.
import '@/components/constellations/styles.css';

export const metadata: Metadata = {
    title: 'ArxiGraph',
    description: 'Visualize mathematical dependency graphs extracted from arXiv papers.',
};

// We don't parse each paper's LaTeX preamble, so paper-specific macros
// (e.g. \F, \G, ...) are unknown. Instead of guessing meanings, we map
// single-letter uppercase macros to their literal letter.
//
// NOTE: This is intentionally conservative. If later you want better fidelity,
// we'd need to extract and feed preamble macros from the source.
const DEFAULT_MATHJAX_CONFIG = `window.MathJax = {
  tex: {
    inlineMath: [['$','$'], ['\\\\(','\\\\)']],
    displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
    processEscapes: true,
    macros: {
      A: 'A', B: 'B', C: 'C', D: 'D', E: 'E', F: 'F', G: 'G', H: 'H', I: 'I', J: 'J',
      K: 'K', L: 'L', M: 'M', N: 'N', O: 'O', P: 'P', Q: 'Q', R: 'R', S: 'S', T: 'T',
      U: 'U', V: 'V', W: 'W', X: 'X', Y: 'Y', Z: 'Z',
      eps: '{\\\\varepsilon}',
    },
  },
};`;

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
                <link
                    href="https://fonts.googleapis.com/css2?family=Inter:wght@700;900&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap"
                    rel="stylesheet"
                />

                {/* MathJax v3 (TeX + CHTML) */}
                <Script
                    id="mathjax-config"
                    strategy="beforeInteractive"
                    dangerouslySetInnerHTML={{ __html: DEFAULT_MATHJAX_CONFIG }}
                />
                <Script
                    id="mathjax-src"
                    strategy="beforeInteractive"
                    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
                />
            </head>
            <body className="antialiased">{children}</body>
        </html>
    );
}
