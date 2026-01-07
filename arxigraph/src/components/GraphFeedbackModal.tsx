'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

type FeedbackScope = 'graph' | 'node' | 'edge';

type FeedbackIssueType =
    | 'missing_node'
    | 'spurious_node'
    | 'wrong_node_type'
    | 'wrong_node_text'
    | 'missing_edge'
    | 'spurious_edge'
    | 'wrong_edge_direction'
    | 'wrong_edge_type'
    | 'graph_missing_section'
    | 'graph_low_coverage'
    | 'graph_layout_confusing'
    | 'other';

type Props = {
    open: boolean;
    onClose: () => void;

    paperId: string;
    graphVersion?: string;

    scope: FeedbackScope;
    nodeId?: string;
    edgeId?: string;

    contextLabel?: string;
};

function issueTypeOptions(scope: FeedbackScope): Array<{ value: FeedbackIssueType; label: string }> {
    if (scope === 'node') {
        return [
            { value: 'spurious_node', label: 'Spurious artifact (should not exist)' },
            { value: 'wrong_node_type', label: 'Wrong type / label' },
            { value: 'wrong_node_text', label: 'Wrong or incomplete statement' },
            { value: 'other', label: 'Other' },
        ];
    }

    if (scope === 'edge') {
        return [
            { value: 'missing_edge', label: 'Missing dependency' },
            { value: 'spurious_edge', label: 'Spurious dependency' },
            { value: 'wrong_edge_direction', label: 'Wrong direction' },
            { value: 'wrong_edge_type', label: 'Wrong dependency type' },
            { value: 'other', label: 'Other' },
        ];
    }

    return [
        { value: 'graph_low_coverage', label: 'Graph is incomplete / too sparse / too coarse' },
        { value: 'graph_layout_confusing', label: 'Layout confusing / misleading' },
        { value: 'other', label: 'Other' },
    ];
}

export default function GraphFeedbackModal(props: Props) {
    const { open, onClose, paperId, graphVersion, scope, nodeId, edgeId, contextLabel } = props;

    const createdAtRef = useRef<number>(Date.now());

    const [issueType, setIssueType] = useState<FeedbackIssueType>('other');
    const [description, setDescription] = useState('');
    const [userDisplay, setUserDisplay] = useState('');
    const [userEmail, setUserEmail] = useState('');

    const [isSubmitting, setIsSubmitting] = useState(false);
    const [success, setSuccess] = useState(false);

    const options = useMemo(() => issueTypeOptions(scope), [scope]);

    useEffect(() => {
        if (!open) return;

        // Reset form every time we open.
        createdAtRef.current = Date.now();
        setIssueType(options[0]?.value ?? 'other');
        setDescription('');
        setUserDisplay('');
        setUserEmail('');
        setIsSubmitting(false);
        setSuccess(false);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open, scope, nodeId, edgeId]);

    if (!open) return null;

    const title =
        scope === 'node'
            ? `Suggest a correction for this node`
            : scope === 'edge'
                ? `Suggest a correction (dependency)`
                : `Suggest a correction`;

    const contextLine =
        scope === 'edge' ? `About edge: ${contextLabel ?? edgeId ?? ''}` : null;

    async function submit() {
        setIsSubmitting(true);

        const clientSubmitMs = Date.now() - createdAtRef.current;

        try {
            const res = await fetch('/api/graph-feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    paperId,
                    graphVersion,
                    scope,
                    nodeId,
                    edgeId,
                    issueType,
                    description: description.trim() ? description.trim() : undefined,
                    userDisplay: userDisplay.trim() ? userDisplay.trim() : undefined,
                    userEmail: userEmail.trim() ? userEmail.trim() : undefined,
                    payload: {
                        contextLabel,
                    },
                    honeypot: '',
                    clientSubmitMs,
                }),
            });

            if (!res.ok && res.status !== 204) {
                const json = (await res.json().catch(() => null)) as any;
                // We don't show infra errors (e.g. Supabase) to the user.
                // Still log so it can be caught in monitoring.
                console.error('graph-feedback submission failed', {
                    status: res.status,
                    error: json?.error,
                    details: json?.details,
                });
            }

            setSuccess(true);
            // Close shortly after success so users see confirmation.
            setTimeout(() => onClose(), 650);
        } catch (e: any) {
            // Always show a positive confirmation in the UI.
            console.error('graph-feedback submission error', e);
            setSuccess(true);
            setTimeout(() => onClose(), 650);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <div
            role="dialog"
            aria-modal="true"
            className="fixed inset-0 z-50 flex items-center justify-center"
        >
            <div
                className="absolute inset-0"
                style={{ background: 'rgba(0,0,0,0.65)' }}
                onClick={() => (isSubmitting ? null : onClose())}
            />

            <div
                className="relative w-[92vw] max-w-xl rounded-xl p-4 sm:p-5"
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
                            {title}
                        </h2>
                        {contextLine ? (
                            <div
                                className="mt-1 text-sm"
                                style={{ color: 'var(--secondary-text)' }}
                            >
                                {contextLine}
                            </div>
                        ) : null}
                    </div>

                    <button
                        type="button"
                        className="paper-link-btn"
                        aria-label="Close"
                        onClick={() => (isSubmitting ? null : onClose())}
                        title="Close"
                    >
                        ×
                    </button>
                </div>

                <div className="mt-4 grid gap-3">
                    <label className="grid gap-1">
                        <span className="text-sm" style={{ color: 'var(--secondary-text)' }}>
                            What is wrong?
                        </span>
                        <select
                            value={issueType}
                            onChange={(e) => setIssueType(e.target.value as FeedbackIssueType)}
                            className="w-full rounded-lg px-3 py-2"
                            style={{
                                background: 'var(--surface2)',
                                border: '1px solid var(--border-color)',
                                color: 'var(--primary-text)',
                            }}
                        >
                            {options.map((o) => (
                                <option key={o.value} value={o.value}>
                                    {o.label}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label className="grid gap-1">
                        <span className="text-sm" style={{ color: 'var(--secondary-text)' }}>
                            Details (optional)
                        </span>
                        <textarea
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            rows={4}
                            placeholder="Briefly explain what’s wrong or what should be there…"
                            className="w-full rounded-lg px-3 py-2"
                            style={{
                                background: 'var(--surface2)',
                                border: '1px solid var(--border-color)',
                                color: 'var(--primary-text)',
                                resize: 'vertical',
                            }}
                        />
                    </label>

                    <div className="grid gap-3 sm:grid-cols-2">
                        <label className="grid gap-1">
                            <span className="text-sm" style={{ color: 'var(--secondary-text)' }}>
                                Name/handle (optional)
                            </span>
                            <input
                                value={userDisplay}
                                onChange={(e) => setUserDisplay(e.target.value)}
                                className="w-full rounded-lg px-3 py-2"
                                style={{
                                    background: 'var(--surface2)',
                                    border: '1px solid var(--border-color)',
                                    color: 'var(--primary-text)',
                                }}
                            />
                        </label>

                        <label className="grid gap-1">
                            <span className="text-sm" style={{ color: 'var(--secondary-text)' }}>
                                Email (optional)
                            </span>
                            <input
                                value={userEmail}
                                onChange={(e) => setUserEmail(e.target.value)}
                                className="w-full rounded-lg px-3 py-2"
                                style={{
                                    background: 'var(--surface2)',
                                    border: '1px solid var(--border-color)',
                                    color: 'var(--primary-text)',
                                }}
                            />
                        </label>
                    </div>

                    <div className="text-xs" style={{ color: 'var(--secondary-text)' }}>
                        This project is experimental. Your feedback helps improve the extraction.
                    </div>

                    {success ? (
                        <div className="text-sm" style={{ color: '#7bed9f' }}>
                            Feedback sent.
                        </div>
                    ) : null}

                    <div className="mt-1 flex items-center justify-end gap-2">
                        <button
                            type="button"
                            onClick={() => (isSubmitting ? null : onClose())}
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
                            onClick={() => (isSubmitting ? null : submit())}
                            disabled={isSubmitting}
                            className="rounded-lg px-4 py-2 font-semibold"
                            style={{
                                background: 'var(--accent)',
                                color: '#111',
                                opacity: isSubmitting ? 0.7 : 1,
                            }}
                        >
                            {isSubmitting ? 'Submitting…' : 'Submit'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
