import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import crypto from 'crypto';

import { getSupabaseAdmin } from '@/lib/supabaseAdmin';

export const runtime = 'nodejs';

const IssueTypeSchema = z.enum([
    'missing_node',
    'spurious_node',
    'wrong_node_type',
    'wrong_node_text',

    'missing_edge',
    'spurious_edge',
    'wrong_edge_direction',
    'wrong_edge_type',

    'graph_missing_section',
    'graph_low_coverage',
    'graph_layout_confusing',

    'other',
]);

const BodySchema = z
    .object({
        paperId: z.string().min(1),
        graphVersion: z.string().min(1).optional(),

        scope: z.enum(['graph', 'node', 'edge']),
        nodeId: z.string().min(1).optional(),
        edgeId: z.string().min(1).optional(),

        issueType: IssueTypeSchema,
        payload: z.record(z.string(), z.unknown()).optional(),

        description: z.string().max(2000).optional(),
        userDisplay: z.string().max(200).optional(),
        userEmail: z.string().email().max(320).optional(),

        // Honeypot field: bots tend to fill it; humans won't see it.
        honeypot: z.string().optional(),

        // Another anti-bot field: time-to-submit in ms (client-provided)
        // We'll just store it in payload for now.
        clientSubmitMs: z.number().int().nonnegative().optional(),
    })
    .superRefine((v, ctx) => {
        if (v.scope === 'node' && !v.nodeId) {
            ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'nodeId required for scope=node' });
        }
        if (v.scope === 'edge' && !v.edgeId) {
            ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'edgeId required for scope=edge' });
        }
    });

function getClientIp(req: NextRequest): string {
    // Works behind proxies if your host sets x-forwarded-for.
    const xff = req.headers.get('x-forwarded-for');
    if (xff) return xff.split(',')[0].trim();
    return '';
}

function hashIp(ip: string): string | null {
    if (!ip) return null;

    const salt = process.env.GRAPH_FEEDBACK_IP_SALT;
    if (!salt) {
        // If not configured, avoid storing raw IP.
        return null;
    }

    return crypto.createHash('sha256').update(`${salt}:${ip}`).digest('hex');
}

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const parsed = BodySchema.safeParse(body);

        if (!parsed.success) {
            return NextResponse.json(
                { error: 'Invalid request', details: parsed.error.flatten() },
                { status: 400 },
            );
        }

        // Honeypot check
        if (parsed.data.honeypot && parsed.data.honeypot.trim().length > 0) {
            // Silently accept to avoid letting bots learn.
            return new NextResponse(null, { status: 204 });
        }

        const ipHash = hashIp(getClientIp(req));
        const userAgent = req.headers.get('user-agent') ?? null;

        let supabase;
        try {
            supabase = getSupabaseAdmin();
        } catch (e: any) {
            return NextResponse.json(
                {
                    error:
                        e?.message ??
                        'Supabase is not configured (set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY).',
                },
                { status: 503 },
            );
        }

        const payload = {
            ...(parsed.data.payload ?? {}),
            clientSubmitMs: parsed.data.clientSubmitMs,
        };

        const insert = {
            paper_id: parsed.data.paperId,
            graph_version: parsed.data.graphVersion ?? null,
            scope: parsed.data.scope,
            node_id: parsed.data.nodeId ?? null,
            edge_id: parsed.data.edgeId ?? null,
            issue_type: parsed.data.issueType,
            payload,
            description: parsed.data.description ?? null,
            user_display: parsed.data.userDisplay ?? null,
            user_email: parsed.data.userEmail ?? null,
            created_by_ip_hash: ipHash,
            user_agent: userAgent,
        };

        const { data, error } = await supabase
            .from('graph_feedback')
            .insert(insert)
            .select('id')
            .single();

        if (error) {
            return NextResponse.json({ error: error.message }, { status: 500 });
        }

        return NextResponse.json({ id: data.id }, { status: 201 });
    } catch (e: any) {
        return NextResponse.json({ error: e?.message ?? String(e) }, { status: 500 });
    }
}
