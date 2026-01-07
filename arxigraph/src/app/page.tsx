'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';

const EXPERIMENTAL_PREVIEW_SEEN_KEY = 'arxigraph.experimentalPreview.seen';

function FlagIcon({ size = 14 }: { size?: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ display: 'inline-block', verticalAlign: 'text-bottom' }}
    >
      <path d="M4 22V4" />
      <path d="M4 4h12l-1.5 4L20 12H4" />
    </svg>
  );
}

function extractArxivId(input: string): string | null {
  // Accept either a full arXiv URL or a bare arXiv identifier.
  // Supports:
  // - https://arxiv.org/abs/2211.11689v1
  // - https://arxiv.org/pdf/2211.11689v1.pdf
  // - old-style: https://arxiv.org/abs/hep-th/9901001
  // - bare ID: 2211.11689v1
  // - bare old-style ID: hep-th/9901001
  const match = input.match(/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+\/\d{7}(?:v\d+)?)/i);
  return match ? match[1] : null;
}

export default function Home() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [enrichContent, setEnrichContent] = useState(true);
  const [inferDependencies, setInferDependencies] = useState(true);

  const [experimentalOpen, setExperimentalOpen] = useState(false);
  const [pendingUrl, setPendingUrl] = useState<string | null>(null);

  const experimentalBody = useMemo(
    () => (
      <>
        <div className="text-sm" style={{ color: 'var(--primary-text)' }}>
          This is a research prototype. Some relationships may be incomplete or incorrectly inferred.
          If you spot an issue, we’d love your feedback{' '}
          <span
            title="Suggest a correction"
            aria-label="Suggest a correction"
            style={{ color: 'var(--secondary-text)' }}
          >
            <FlagIcon size={14} />
          </span>
          .
        </div>
      </>
    ),
    [],
  );

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const formData = new FormData(event.currentTarget);
    const input = (formData.get('arxivUrl') as string).trim();
    // Use component state; the toggles are clickable pills (no checkbox inputs).

    const arxivId = extractArxivId(input);
    if (!arxivId) {
      setError(
        'Could not extract an arXiv ID from the provided input. Please enter a full arXiv URL (https://arxiv.org/abs/...) or an arXiv identifier like 1410.5929v1.',
      );
      return;
    }

    const search = new URLSearchParams();
    search.set('enrich', enrichContent ? '1' : '0');
    search.set('deps', inferDependencies ? '1' : '0');

    const url = `/paper/${encodeURIComponent(arxivId)}?${search.toString()}`;

    let hasSeenExperimental = true;
    try {
      // Only runs in the browser.
      hasSeenExperimental = localStorage.getItem(EXPERIMENTAL_PREVIEW_SEEN_KEY) === '1';
    } catch {
      // If localStorage isn't available, default to "seen" to avoid blocking.
      hasSeenExperimental = true;
    }

    if (!hasSeenExperimental) {
      setPendingUrl(url);
      setExperimentalOpen(true);
      return;
    }

    router.push(url);
  };

  const handleExperimentalCancel = () => {
    setExperimentalOpen(false);
    setPendingUrl(null);
  };

  const handleExperimentalContinue = () => {
    try {
      localStorage.setItem(EXPERIMENTAL_PREVIEW_SEEN_KEY, '1');
    } catch {
      // ignore
    }
    setExperimentalOpen(false);
    if (pendingUrl) router.push(pendingUrl);
    setPendingUrl(null);
  };

  return (
    <main
      className="flex flex-col items-center min-h-screen p-4 sm:p-8 md:p-12"
      style={{ background: 'var(--background)', color: 'var(--primary-text)' }}
    >
      {experimentalOpen ? (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center"
        >
          <div
            className="absolute inset-0"
            style={{ background: 'rgba(0,0,0,0.65)' }}
            onClick={handleExperimentalCancel}
          />

          <div
            className="relative w-[92vw] max-w-lg rounded-xl p-4 sm:p-5"
            style={{
              background: 'var(--surface1)',
              border: '1px solid var(--border-color)',
              color: 'var(--primary-text)',
            }}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2
                  className="text-lg font-semibold"
                  style={{ fontFamily: 'Inter, system-ui, sans-serif' }}
                >
                  Experimental preview
                </h2>
              </div>

              <button
                type="button"
                className="paper-link-btn"
                aria-label="Close"
                onClick={handleExperimentalCancel}
                title="Close"
              >
                ×
              </button>
            </div>

            <div className="mt-4 grid gap-4">
              {experimentalBody}

              <div className="mt-1 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={handleExperimentalCancel}
                  className="rounded-lg px-3 py-2"
                  style={{
                    border: '1px solid var(--border-color)',
                    color: 'var(--secondary-text)',
                  }}
                >
                  Cancel
                </button>

                <button
                  type="button"
                  onClick={handleExperimentalContinue}
                  className="rounded-lg px-4 py-2 font-semibold"
                  style={{
                    background: 'var(--accent)',
                    color: '#111',
                  }}
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <div className="w-full max-w-4xl text-center">
        <h1
          className="text-4xl sm:text-5xl font-black tracking-tight"
          style={{ color: 'var(--accent)' }}
        >
          ArxiGraph
        </h1>
        <p className="mt-2 text-lg" style={{ color: 'var(--secondary-text)' }}>
          Visualize the logical structure of mathematical papers on arXiv.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="w-full max-w-xl mt-8 flex flex-col gap-3"
        style={{
          background: 'var(--surface1)',
          border: '1px solid var(--border-color)',
          borderRadius: 12,
          padding: 16,
        }}
      >
        <div className="relative">
          <input
            type="text"
            name="arxivUrl"
            placeholder="https://arxiv.org/abs/... or 1410.5929v1"
            required
            className="w-full p-3 pr-12 rounded-lg shadow-sm focus:outline-none"
            style={{
              background: 'var(--surface2)',
              border: '1px solid var(--border-color)',
              color: 'var(--primary-text)',
            }}
          />

          <button
            type="submit"
            aria-label="Generate graph"
            title="Generate graph"
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-md transition-colors hover:text-[var(--accent)] focus-visible:text-[var(--accent)] focus-visible:outline-none"
            style={{
              color: 'var(--secondary-text)',
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              focusable="false"
            >
              <path d="M7 17L17 7" />
              <path d="M7 7h10v10" />
            </svg>
          </button>
        </div>

        <div
          className="mt-2 flex flex-wrap items-center justify-center gap-3 text-xs sm:text-sm"
          style={{ color: 'var(--secondary-text)' }}
        >
          <button
            type="button"
            onClick={() => setEnrichContent((v) => !v)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-full border cursor-pointer transition"
            style={{
              borderColor: enrichContent ? 'var(--accent)' : 'var(--border-color)',
              background: enrichContent ? 'var(--surface1)' : 'var(--surface2)',
              color: 'var(--secondary-text)',
            }}
            aria-pressed={enrichContent}
          >
            <span suppressHydrationWarning>
              Enhance artifacts (definitions/symbols)
            </span>
          </button>

          <button
            type="button"
            onClick={() => setInferDependencies((v) => !v)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-full border cursor-pointer transition"
            style={{
              borderColor: inferDependencies ? 'var(--accent)' : 'var(--border-color)',
              background: inferDependencies ? 'var(--surface1)' : 'var(--surface2)',
              color: 'var(--secondary-text)',
            }}
            aria-pressed={inferDependencies}
          >
            Infer dependencies
          </button>
        </div>
      </form>

      {error && (
        <p className="mt-4 font-semibold" style={{ color: '#ff6b6b' }}>
          Error: {error}
        </p>
      )}
    </main>
  );
}
