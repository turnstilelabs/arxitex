/* eslint-disable @next/next/no-page-custom-font */

import type { Metadata } from 'next';
import Script from 'next/script';

import './globals.css';
// Constellations global styles (kept in components so graph UI can share it).
// Importing here avoids CSS `@import` path resolution issues in Turbopack.
import '@/components/constellations/styles.css';

// Default MathJax config. Per-paper macros extracted from the LaTeX preamble
// can be merged into `window.MathJax.tex.macros` at runtime by the page
// components (see paper/[arxivId]/page.tsx).
const DEFAULT_MATHJAX_CONFIG = `window.MathJax = {
  tex: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']],
    processEscapes: true,
    macros: {
      eps: '{\\varepsilon}',
    },
  },
};`;

export const metadata: Metadata = {
    title: 'ArxiGraph',
    description: 'Visualize mathematical dependency graphs extracted from arXiv papers.',
};

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
