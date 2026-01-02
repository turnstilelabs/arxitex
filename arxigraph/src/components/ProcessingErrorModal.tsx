"use client";

export type ProcessingError = {
    code: string;
    message: string;
    stage?: string;
};

type Props = {
    open: boolean;
    error: ProcessingError | null;
    onClose: () => void;
    onRetry?: () => void;
    currentArxivUrl?: string;
};

type ErrorCopy = {
    title: string;
    description: string;
    primaryActionLabel?: string;
    secondaryActionLabel?: string;
    allowRetry?: boolean;
};

const ERROR_COPY: Record<string, ErrorCopy> = {
    no_latex_source: {
        title: "This paper doesn’t have LaTeX source",
        description:
            "We can only build a dependency graph when arXiv provides the LaTeX source. This paper is available as PDF only, so we can’t extract its internal structure.",
    },
    graph_empty: {
        title: "No mathematical statements detected",
        description:
            "We processed the LaTeX source but couldn’t find theorems, lemmas, or similar statements to build a graph from. This sometimes happens for short notes, surveys, or non-technical content.",
    },
    enhancements_misconfigured: {
        title: "Enhancements are not configured on the server",
        description:
            "Enhancements were requested, but the server is not configured with an API key for the language model. You can disable enhancements or contact the maintainer to set it up.",
    },
    invalid_arxiv_id: {
        title: "This doesn’t look like a valid arXiv link",
        description:
            "The arXiv identifier looks invalid. Please use a URL like https://arxiv.org/abs/2211.11689 or a standard arXiv ID.",
    },
    source_download_failed: {
        title: "Couldn’t download the LaTeX source",
        description:
            "We tried several times to download the LaTeX source from arXiv but couldn’t complete the download. This might be a temporary network or arXiv issue.",
        primaryActionLabel: "Retry",
        allowRetry: true,
    },
    source_gzip_corrupt: {
        title: "The source archive from arXiv looks corrupted",
        description:
            "We downloaded the LaTeX source archive from arXiv but couldn’t open it. The file appears to be corrupted or in an unexpected format.",
        primaryActionLabel: "Try another paper",
    },
    source_tar_corrupt: {
        title: "The source archive from arXiv looks corrupted",
        description:
            "We downloaded the LaTeX source archive from arXiv but couldn’t open it. The file appears to be corrupted or in an unexpected format.",
        primaryActionLabel: "Try another paper",
    },
    source_zip_corrupt: {
        title: "The source archive from arXiv looks corrupted",
        description:
            "We downloaded the LaTeX source archive from arXiv but couldn’t open it. The file appears to be corrupted or in an unexpected format.",
        primaryActionLabel: "Try another paper",
    },
    source_extract_failed: {
        title: "We couldn’t unpack the source files",
        description:
            "We downloaded the LaTeX source from arXiv but weren’t able to unpack it into usable files. The archive format looks unsupported or malformed.",
        primaryActionLabel: "Try another paper",
    },
    llm_rate_limited: {
        title: "The language model is temporarily rate-limited",
        description:
            "The language model provider is currently rate-limiting requests. Please wait a bit and try again.",
        primaryActionLabel: "Retry",
        allowRetry: true,
    },
    llm_timeout: {
        title: "The language model took too long to respond",
        description:
            "The language model didn’t respond in time while we were building the graph. You can retry, or disable enhancements for a faster but less detailed graph.",
        primaryActionLabel: "Retry",
        allowRetry: true,
    },
    llm_connection_error: {
        title: "We couldn’t reach the language model service",
        description:
            "There was a network or HTTP error while talking to the language model. This is usually temporary. Please try again in a few minutes.",
        primaryActionLabel: "Retry",
        allowRetry: true,
    },
    llm_api_error: {
        title: "The language model returned an error",
        description:
            "The language model provider returned an error while processing this paper. This is likely a temporary issue with the service.",
        primaryActionLabel: "Retry",
        allowRetry: true,
    },
    backend_unreachable: {
        title: "The processing backend is not reachable",
        description:
            "We couldn’t reach the processing backend service. It may be restarting or temporarily offline. Please try again in a moment.",
        primaryActionLabel: "Retry",
        allowRetry: true,
    },
    unexpected_error: {
        title: "Something went wrong while processing this paper",
        description:
            "An unexpected error occurred on our side. The details have been logged so we can investigate.",
    },
    // Fallback for any unknown code
    default: {
        title: "Something went wrong while processing this paper",
        description:
            "An unexpected error occurred while processing this paper. The details have been logged so we can investigate.",
        primaryActionLabel: "Close",
    },
};

export default function ProcessingErrorModal(props: Props) {
    const { open, error, onClose, onRetry, currentArxivUrl } = props;

    if (!open || !error) return null;

    const meta = ERROR_COPY[error.code] ?? ERROR_COPY.default;

    const handlePrimaryClick = () => {
        if (meta.allowRetry && onRetry) {
            onRetry();
            return;
        }
        onClose();
    };

    const primaryLabel = meta.allowRetry ? (meta.primaryActionLabel ?? "Retry") : undefined;
    const showPrimary = !!(meta.allowRetry && primaryLabel && onRetry);

    return (
        <div
            role="dialog"
            aria-modal="true"
            className="fixed inset-0 z-50 flex items-center justify-center"
        >
            <div
                className="absolute inset-0"
                style={{ background: "rgba(0,0,0,0.65)" }}
                onClick={onClose}
            />

            <div
                className="relative w-[92vw] max-w-xl rounded-xl p-4 sm:p-5"
                style={{
                    background: "var(--surface1)",
                    border: "1px solid var(--border-color)",
                    color: "var(--primary-text)",
                }}
            >
                <div className="flex items-start justify-between gap-3">
                    <div>
                        <h2
                            className="text-lg font-semibold"
                            style={{ fontFamily: "Inter, system-ui, sans-serif" }}
                        >
                            {meta.title}
                        </h2>
                        {currentArxivUrl ? (
                            <div
                                className="mt-1 text-xs sm:text-sm"
                                style={{ color: "var(--secondary-text)" }}
                            >
                                While processing: {currentArxivUrl}
                            </div>
                        ) : null}
                    </div>

                    <button
                        type="button"
                        className="paper-link-btn"
                        aria-label="Close"
                        onClick={onClose}
                        title="Close"
                    >
                        ×
                    </button>
                </div>

                <div className="mt-4 grid gap-3 text-sm">
                    <p style={{ color: "var(--primary-text)" }}>{meta.description}</p>

                    {showPrimary ? (
                        <div className="mt-2 flex items-center justify-end gap-2">
                            <button
                                type="button"
                                onClick={handlePrimaryClick}
                                className="rounded-lg px-4 py-2 font-semibold"
                                style={{
                                    background: "var(--accent)",
                                    color: "#111",
                                }}
                            >
                                {primaryLabel}
                            </button>
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    );
}
