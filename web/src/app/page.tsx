"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getLlmStatus } from "@/lib/api";
import type { Job } from "@/lib/types";
import Logo from "@/components/Logo";

export default function HomePage() {
  const router = useRouter();
  const [arxivUrlOrId, setArxivUrlOrId] = useState("");
  const [inferDependencies, setInferDependencies] = useState(false);
  const [enrichContent, setEnrichContent] = useState(false);
  const [force, setForce] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

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
    setSubmitting(true);

    const id = arxivUrlOrId.trim();
    // Always navigate immediately and let the paper page stream the build via SSE.
    const qs = new URLSearchParams({
      stream: "1",
      infer: String(inferDependencies),
      enrich: String(enrichContent),
      force: String(force),
    });
    router.push(`/papers/${id}?${qs.toString()}`);
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-6 md:px-12">
      <div className="w-full max-w-[600px] mx-auto text-center space-y-10 md:space-y-14">
        <div className="flex justify-center mt-6 mb-3">
          <Logo className="w-[220px] h-auto" withText={true} />
        </div>
        <p className="text-slate-600 mb-12 max-w-2xl mx-auto text-center text-xl md:text-2xl leading-relaxed tracking-[2px]">
          The shortest path to understanding research
        </p>

        <form onSubmit={handleSubmit} className="space-y-5 bg-[#f9f9f9] rounded-xl ring-1 ring-slate-200 p-8 md:p-10 shadow-[0_4px_8px_rgba(0,0,0,0.10)] max-w-[600px] mx-auto">
          <div>
            <div className="relative">
              <input
                id="paper-source"
                className="w-full border rounded-md px-3 py-2 pr-24 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 placeholder:text-slate-400"
                placeholder="Enter arXiv URL or ID (e.g. 2211.11689)"
                value={arxivUrlOrId}
                onChange={(e) => setArxivUrlOrId(e.target.value)}
                required
              />
              <div className="absolute inset-y-0 right-2 z-10 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowAdvanced((s) => !s)}
                  className="text-slate-400 hover:text-slate-600"
                  aria-expanded={showAdvanced}
                  aria-controls="advanced-options"
                  title="Advanced options"
                  aria-label="Advanced options"
                >
                  {/* Gear icon */}
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0A1.65 1.65 0 0 0 10.91 3H11a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0A1.65 1.65 0 0 0 21 10.91V11a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                  </svg>
                </button>
                <button
                  type="submit"
                  className="h-9 w-9 rounded-full bg-[#007bff] text-white hover:bg-[#0069d9] focus:outline-none focus:ring-2 focus:ring-[#007bff] disabled:opacity-60 inline-flex items-center justify-center"
                  disabled={!arxivUrlOrId || submitting}
                  aria-label="Analyze paper"
                  title="Analyze paper"
                >
                  {submitting ? (
                    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0 -4 4H4z"></path>
                    </svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <path d="M8 5v14l11-7z"></path>
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>



          {showAdvanced && (
            <div id="advanced-options" className="mt-3 flex flex-col gap-3">
              <label
                className="inline-flex items-center gap-2"
                title={llmAvailable === false ? "Infer missing artifact links using AI (unavailable on server)" : "Infer missing artifact links using AI"}
              >
                <input
                  type="checkbox"
                  checked={inferDependencies}
                  onChange={(e) => setInferDependencies(e.target.checked)}
                  disabled={llmAvailable === false}
                />
                <span className="text-sm text-[var(--text)]">Infer missing links using AI</span>
              </label>

              <label
                className="inline-flex items-center gap-2"
                title={llmAvailable === false ? "Add definitions automatically (unavailable on server)" : "Add definitions automatically"}
              >
                <input
                  type="checkbox"
                  checked={enrichContent}
                  onChange={(e) => setEnrichContent(e.target.checked)}
                  disabled={llmAvailable === false}
                />
                <span className="text-sm text-[var(--text)]">Add definitions automatically</span>
              </label>

              <label className="inline-flex items-center gap-2" title="Re-process the paper even if it already exists">
                <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
                <span className="text-sm text-[var(--text)]">Re-analyze existing paper</span>
              </label>
            </div>
          )}
          {job && (
            <div className="mt-2 text-sm text-gray-600">
              Status: {job.status} · Stage: {job.stage} · {job.progress}%
            </div>
          )}

          {error && <div className="text-sm text-red-600">{error}</div>}

          <div className="text-xs text-gray-500 pt-2">
            {llmAvailable === false && (
              <div className="text-xs text-red-600 mt-1">
                LLM features are not available on the server. Set OPENAI_API_KEY or TOGETHER_API_KEY to enable them.
              </div>
            )}
          </div>
        </form>
        <p className="mt-3 text-xs text-slate-500 text-center">
          Try an example →{" "}
          <button
            type="button"
            className="underline decoration-dotted hover:text-slate-700"
            onClick={() => setArxivUrlOrId("2211.11689")}
          >
            2211.11689
          </button>
        </p>
      </div >
    </main >
  );
}
