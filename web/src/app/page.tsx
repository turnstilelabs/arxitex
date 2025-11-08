"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ingestPaper, getJob, getLlmStatus } from "@/lib/api";
import type { IngestRequest, Job } from "@/lib/types";
import Logo from "@/components/Logo";

export default function HomePage() {
  const router = useRouter();
  const [arxivUrlOrId, setArxivUrlOrId] = useState("");
  const [inferDependencies, setInferDependencies] = useState(false);
  const [enrichContent, setEnrichContent] = useState(false);
  const [force, setForce] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [llmAvailable, setLlmAvailable] = useState<boolean | null>(null);
  const [llmProviders, setLlmProviders] = useState<string[]>([]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await getLlmStatus();
        if (!mounted) return;
        setLlmAvailable(res.available);
        setLlmProviders(res.providers || []);
      } catch (e) {
        if (!mounted) return;
        setLlmAvailable(false);
        setLlmProviders([]);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const payload: IngestRequest = {
      arxiv_url_or_id: arxivUrlOrId.trim(),
      infer_dependencies: inferDependencies,
      enrich_content: enrichContent,
      force,
    };

    try {
      setSubmitting(true);
      const j = await ingestPaper(payload);
      setJob(j);

      // Poll job until done/failed
      let attempts = 0;
      const maxAttempts = 120; // ~2 minutes at 1s interval

      while (true) {
        attempts++;
        const cur = await getJob(j.job_id);
        setJob(cur);

        if (cur.status === "done") {
          // Navigate to paper page
          router.push(`/papers/${cur.arxiv_id}`);
          break;
        }
        if (cur.status === "failed") {
          setError(cur.error || "Processing failed");
          break;
        }
        if (attempts >= maxAttempts) {
          setError("Timed out waiting for processing to complete.");
          break;
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
    } catch (err: any) {
      setError(err?.message || "Failed to start ingestion");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen p-6 md:p-12 bg-gradient-to-b from-slate-50 to-white">
      <div className="max-w-3xl mx-auto">
        <div className="flex justify-center mt-6 mb-3">
          <Logo className="h-12 w-auto" withText={true} />
        </div>
        <p className="text-slate-700 mb-10 max-w-2xl mx-auto text-center text-lg md:text-xl leading-relaxed">
          Explore the structure behind the mathematics.
        </p>

        <form onSubmit={handleSubmit} className="space-y-5 bg-white rounded-xl ring-1 ring-slate-200 p-5 md:p-6 shadow-sm">
          <div>
            <label className="block text-sm font-medium mb-1 text-black">arXiv URL or ID</label>
            <input
              className="w-full border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="e.g. https://arxiv.org/abs/2211.11689 or 2211.11689"
              value={arxivUrlOrId}
              onChange={(e) => setArxivUrlOrId(e.target.value)}
              required
            />
          </div>

          <div className="flex flex-wrap items-center gap-6">
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={inferDependencies}
                onChange={(e) => setInferDependencies(e.target.checked)}
                disabled={llmAvailable === false}
                title={llmAvailable === false ? "LLM features are not available on the server" : ""}
              />
              <span className="text-sm text-black">Infer dependencies (LLM)</span>
            </label>

            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={enrichContent}
                onChange={(e) => setEnrichContent(e.target.checked)}
                disabled={llmAvailable === false}
                title={llmAvailable === false ? "LLM features are not available on the server" : ""}
              />
              <span className="text-sm text-black">Enrich content with definitions (LLM)</span>
            </label>

            <label className="inline-flex items-center gap-2">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
              <span className="text-sm text-black">Force re-process</span>
            </label>
          </div>


          <div className="flex items-center gap-3">
            <button
              type="submit"
              className="px-4 py-2 rounded-md bg-indigo-600 text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60 inline-flex items-center gap-2"
              disabled={!arxivUrlOrId || submitting}
            >
              {submitting ? (
                <>
                  <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
                  </svg>
                  Processing…
                </>
              ) : (
                "Process"
              )}
            </button>
            {job && (
              <span className="text-sm text-gray-600">
                Status: {job.status} · Stage: {job.stage} · {job.progress}%
              </span>
            )}
          </div>

          {error && <div className="text-sm text-red-600">{error}</div>}

          <div className="text-xs text-gray-500 pt-2">
            {llmAvailable === false && (
              <div className="text-xs text-red-600 mt-1">
                LLM features are not available on the server. Set OPENAI_API_KEY or TOGETHER_API_KEY to enable them.
              </div>
            )}
          </div>
        </form>
      </div>
    </main>
  );
}
