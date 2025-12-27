'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

function extractArxivId(arxivUrl: string): string | null {
  // Supports:
  // - https://arxiv.org/abs/2211.11689v1
  // - https://arxiv.org/pdf/2211.11689v1.pdf
  // - old-style: https://arxiv.org/abs/hep-th/9901001
  const match = arxivUrl.match(/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+\/\d{7}(?:v\d+)?)/i);
  return match ? match[1] : null;
}

export default function Home() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const formData = new FormData(event.currentTarget);
    const arxivUrl = (formData.get('arxivUrl') as string).trim();
    const enrichContent = formData.get('enrichContent') === 'on';
    const inferDependencies = formData.get('inferDependencies') === 'on';

    const arxivId = extractArxivId(arxivUrl);
    if (!arxivId) {
      setError('Could not extract an arXiv ID from the provided URL.');
      return;
    }

    const search = new URLSearchParams();
    search.set('enrich', enrichContent ? '1' : '0');
    search.set('deps', inferDependencies ? '1' : '0');

    router.push(`/paper/${encodeURIComponent(arxivId)}?${search.toString()}`);
  };

  return (
    <main
      className="flex flex-col items-center min-h-screen p-4 sm:p-8 md:p-12"
      style={{ background: 'var(--background)', color: 'var(--primary-text)' }}
    >
      <div className="w-full max-w-4xl text-center">
        <h1
          className="text-4xl sm:text-5xl font-black tracking-tight"
          style={{ color: 'var(--accent)' }}
        >
          ArxiGraph
        </h1>
        <p className="mt-2 text-lg" style={{ color: 'var(--secondary-text)' }}>
          Paste an arXiv URL to visualize its mathematical dependency graph.
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
        <div className="flex items-center gap-2">
          <input
            type="url"
            name="arxivUrl"
            placeholder="https://arxiv.org/abs/..."
            required
            className="flex-grow p-3 rounded-lg shadow-sm focus:outline-none"
            style={{
              background: 'var(--surface2)',
              border: '1px solid var(--border-color)',
              color: 'var(--primary-text)',
            }}
          />
          <button
            type="submit"
            className="px-6 py-3 font-semibold rounded-lg shadow-md transition-colors"
            style={{
              background: 'var(--accent)',
              color: 'var(--background)',
              border: '2px solid var(--accent)',
            }}
          >
            Generate Graph
          </button>
        </div>

        <div
          className="mt-2 flex flex-wrap items-center justify-center gap-3 text-xs sm:text-sm"
          style={{ color: 'var(--secondary-text)' }}
        >
          <label
            className="inline-flex items-center gap-2 px-3 py-2 rounded-full border border-[var(--border-color)] bg-[var(--surface2)] cursor-pointer transition hover:border-[var(--accent)] hover:bg-[var(--surface1)]"
          >
            <input
              type="checkbox"
              name="enrichContent"
              defaultChecked
              className="h-3 w-3 accent-[var(--accent)]"
            />
            <span className="text-[0.78rem] sm:text-xs md:text-sm">
              Enhance artifacts (definitions/symbols)
            </span>
          </label>

          <label
            className="inline-flex items-center gap-2 px-3 py-2 rounded-full border border-[var(--border-color)] bg-[var(--surface2)] cursor-pointer transition hover:border-[var(--accent)] hover:bg-[var(--surface1)]"
          >
            <input
              type="checkbox"
              name="inferDependencies"
              defaultChecked
              className="h-3 w-3 accent-[var(--accent)]"
            />
            <span className="text-[0.78rem] sm:text-xs md:text-sm">Infer dependencies</span>
          </label>
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
