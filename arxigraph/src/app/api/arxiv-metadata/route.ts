import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';

export const runtime = 'nodejs';

const querySchema = z.object({
    arxivId: z.string().min(1),
});

function stripVersion(arxivId: string) {
    return arxivId.replace(/v\d+$/i, '');
}

function decodeXmlEntities(s: string) {
    return s
        .replace(/&/g, '&')
        .replace(/</g, '<')
        .replace(/>/g, '>')
        .replace(/"/g, '"')
        .replace(/'/g, "'")
        .replace(/&#39;/g, "'")
        .replace(/&#(\d+);/g, (_m, n) => String.fromCharCode(Number(n)));
}

function normalizeText(s: string) {
    return decodeXmlEntities(s).replace(/\s+/g, ' ').trim();
}

function parseAtomEntry(xml: string): { title: string; authors: string[] } | null {
    // arXiv returns an Atom feed. There's a <title> for the <feed> and a <title> for the <entry>.
    // We want the entry title and authors.
    const entryMatch = xml.match(/<entry>([\s\S]*?)<\/entry>/);
    if (!entryMatch) return null;

    const entryXml = entryMatch[1];

    const titleMatch = entryXml.match(/<title[^>]*>([\s\S]*?)<\/title>/);
    const title = titleMatch ? normalizeText(titleMatch[1]) : '';

    const authors = Array.from(
        entryXml.matchAll(
            /<author>[\s\S]*?<name[^>]*>([\s\S]*?)<\/name>[\s\S]*?<\/author>/g,
        ),
    )
        .map((m) => normalizeText(m[1]))
        .filter(Boolean);

    return { title, authors };
}

export async function GET(req: NextRequest) {
    const url = new URL(req.url);
    const parse = querySchema.safeParse({
        arxivId: url.searchParams.get('arxivId') ?? undefined,
    });

    if (!parse.success) {
        return NextResponse.json({ error: 'Invalid arxivId' }, { status: 400 });
    }

    const arxivId = stripVersion(parse.data.arxivId);
    const upstream = `https://export.arxiv.org/api/query?id_list=${encodeURIComponent(arxivId)}`;

    try {
        const res = await fetch(upstream, {
            cache: 'no-store',
            headers: {
                'User-Agent': 'arxigraph/0.1',
            },
        });

        if (!res.ok) {
            return NextResponse.json(
                { error: `Upstream error (status ${res.status})` },
                { status: 502 },
            );
        }

        const xml = await res.text();
        const parsedEntry = parseAtomEntry(xml);

        if (!parsedEntry) {
            return NextResponse.json({ error: 'No arXiv entry found' }, { status: 404 });
        }

        return NextResponse.json({ arxivId, title: parsedEntry.title, authors: parsedEntry.authors });
    } catch (e: any) {
        return NextResponse.json({ error: e?.message ?? String(e) }, { status: 500 });
    }
}
