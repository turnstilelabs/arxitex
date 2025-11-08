import React from "react";

type LogoProps = {
    className?: string;
    withText?: boolean;
    title?: string;
};

export default function Logo({ className = "", withText = true, title = "ArxiTex" }: LogoProps) {
    // Responsive SVG: size is controlled by the parent via className (e.g., h-10 w-auto)
    return (
        <svg
            viewBox="0 0 240 60"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            role="img"
            aria-label={title}
            className={`${className} block align-middle`}
        >
            <title>{title}</title>
            <defs>
                <linearGradient id="arxitex-g" x1="0" y1="0" x2="240" y2="60" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#6366F1" />
                    <stop offset="1" stopColor="#06B6D4" />
                </linearGradient>
                <filter id="arxitex-softShadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="rgba(2,6,23,0.15)" />
                </filter>
            </defs>

            {/* Graph mark */}
            <g filter="url(#arxitex-softShadow)">
                <circle cx="26" cy="16" r="6" fill="url(#arxitex-g)" />
                <circle cx="54" cy="44" r="6" fill="url(#arxitex-g)" />
                <circle cx="26" cy="44" r="6" fill="url(#arxitex-g)" />
                <line x1="26" y1="22" x2="26" y2="38" stroke="url(#arxitex-g)" strokeWidth="3.5" strokeLinecap="round" />
                <line x1="32" y1="44" x2="48" y2="44" stroke="url(#arxitex-g)" strokeWidth="3.5" strokeLinecap="round" />
                <line
                    x1="30"
                    y1="20"
                    x2="50"
                    y2="40"
                    stroke="url(#arxitex-g)"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                    opacity="0.85"
                />
            </g>

            {/* Wordmark (optional) */}
            {withText ? (
                <text
                    x="78"
                    y="40"
                    fill="#0F172A"
                    fontSize="28"
                    fontWeight="700"
                    letterSpacing="0.4"
                    fontFamily='"STIX Two Text", Merriweather, Georgia, "Times New Roman", serif'
                >
                    ArxiTex
                </text>
            ) : null}
        </svg>
    );
}
