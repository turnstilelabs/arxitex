import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { spawn } from 'child_process';

import {
  GraphNode,
  GraphLink,
  PythonGraphResponseSchema,
} from '@/lib/schemas';

// Ensure this route runs in a Node.js runtime so we can spawn Python.
// Deprecated: the frontend now calls the separate Python backend.
// Kept for reference; you can delete this route once you're confident.
export const runtime = 'nodejs';

// Define the schema for the request body
const processPaperSchema = z.object({
  arxivUrl: z.string().url().regex(/arxiv\.org/),
  inferDependencies: z.boolean().optional().default(true),
  enrichContent: z.boolean().optional().default(true),
});

// Helper to stream data back to the client as Server-Sent Events
function streamData(
  controller: ReadableStreamDefaultController<any>,
  type: string,
  data: any,
) {
  const encoder = new TextEncoder();
  controller.enqueue(
    encoder.encode(`data: ${JSON.stringify({ type, data })}\n\n`),
  );
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const validation = processPaperSchema.safeParse(body);

    if (!validation.success) {
      return new NextResponse('Invalid request body', { status: 400 });
    }

    const { arxivUrl, inferDependencies, enrichContent } = validation.data;

    // Extract arXiv ID from URL (supports new-style and old-style IDs).
    const arxivIdMatch = arxivUrl.match(/(\d{4}\.\d{4,5}|[a-z-]+\/\d{7})/i);
    if (!arxivIdMatch) {
      return new NextResponse('Could not extract arXiv ID from URL', {
        status: 400,
      });
    }
    const arxivId = arxivIdMatch[0];

    const stream = new ReadableStream({
      async start(controller) {
        const log = (message: string) =>
          streamData(controller, 'status', message);

        try {
          log('Analysis started using Python pipeline...');
          log(`Extracted arXiv ID: ${arxivId}`);

          // Build Python command: python -m arxitex.tools.graph_json_cli <id> [flags]
          const args = [
            '-m',
            'arxitex.tools.graph_json_cli',
            arxivId,
          ];

          // Only enable enhancements when explicitly requested.
          if (inferDependencies && enrichContent) {
            args.push('--all-enhancements');
          } else {
            if (inferDependencies) args.push('--infer-deps');
            if (enrichContent) args.push('--enrich-content');
          }

          const python = spawn('python', args, {
            cwd: process.cwd(),
          });

          const stdoutChunks: Buffer[] = [];

          python.stdout.on('data', (chunk: Buffer) => {
            stdoutChunks.push(chunk);
          });

          python.stderr.on('data', (chunk: Buffer) => {
            const text = chunk.toString('utf8');
            for (const line of text.split(/\r?\n/)) {
              const trimmed = line.trim();
              if (trimmed) {
                log(trimmed);
              }
            }
          });

          python.on('error', (err: Error) => {
            log(`Failed to start Python process: ${err.message}`);
            controller.close();
          });

          python.on('close', (code: number | null) => {
            (async () => {
              if (code !== 0) {
                log(`Python pipeline exited with code ${code}`);
                controller.close();
                return;
              }

              try {
                const raw = Buffer.concat(stdoutChunks).toString('utf8').trim();
                if (!raw) {
                  log('Python pipeline produced no output.');
                  controller.close();
                  return;
                }

                const json = JSON.parse(raw);
                const parsed = PythonGraphResponseSchema.parse(json);

                const { graph } = parsed;
                log(
                  `Received graph with ${graph.stats.node_count} nodes and ${graph.stats.edge_count} edges.`,
                );

                // Stream nodes
                for (const n of graph.nodes) {
                  const node: GraphNode = {
                    id: n.id,
                    type: (n.type as any),
                    label: (n.label ?? n.display_name ?? n.id) as string,
                    content: n.content,
                    // We use edges to represent explicit dependencies; this
                    // field is kept for compatibility with the existing
                    // GraphNode schema.
                    dependencies: [],
                  };
                  streamData(controller, 'node', node);
                }

                // Stream edges as links
                for (const e of graph.edges) {
                  const link: GraphLink = {
                    source: e.source,
                    target: e.target,
                    dependencyType: e.dependency_type ?? undefined,
                    referenceType: e.reference_type ?? undefined,
                    dependency: e.dependency ?? undefined,
                    context: e.context ?? undefined,
                    edgeType: e.type ?? undefined,
                  };
                  streamData(controller, 'link', link);
                }

                log('Analysis complete.');
              } catch (e: any) {
                log(`Failed to parse Python output: ${e.message}`);
              } finally {
                controller.close();
              }
            })().catch((err) => {
              log(`Unexpected error while handling Python output: ${err}`);
              controller.close();
            });
          });
        } catch (e: any) {
          log(`Error during stream processing: ${e.message}`);
          console.error('Error during stream processing:', e);
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    });
  } catch (error) {
    console.error('API Error:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
