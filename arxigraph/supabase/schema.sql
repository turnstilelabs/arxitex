-- Supabase schema for collecting community feedback about the extracted paper graph.
--
-- Apply this in your Supabase SQL editor.

create extension if not exists pgcrypto;

create table if not exists public.graph_feedback (
  id                 uuid primary key default gen_random_uuid(),

  -- What graph / where
  paper_id           text not null,
  graph_version      text,

  scope              text not null check (scope in ('graph', 'node', 'edge')),
  node_id            text,
  edge_id            text,

  issue_type          text not null,
  payload             jsonb,
  description         text,

  -- Identity (auth comes later; keep ready)
  created_by_user_id  uuid,
  user_display        text,
  user_email          text,

  -- Anti-spam / heuristics
  created_by_ip_hash  text,
  user_agent          text,

  -- Resolution
  status              text not null default 'pending' check (status in ('pending', 'accepted', 'rejected', 'duplicate', 'spam')),
  resolution_note     text,
  resolved_at         timestamptz,

  created_at          timestamptz not null default now()
);

create index if not exists graph_feedback_paper_id_idx on public.graph_feedback (paper_id);
create index if not exists graph_feedback_status_idx on public.graph_feedback (status);
create index if not exists graph_feedback_created_at_idx on public.graph_feedback (created_at);

-- Optional guardrails (not strictly required):
-- Ensure node_id is present when scope='node' and edge_id is present when scope='edge'.
-- Note: CHECK constraints can be too strict during early iteration, so you can remove
-- them if they get in the way.
alter table public.graph_feedback
  drop constraint if exists graph_feedback_scope_requires_target;

alter table public.graph_feedback
  add constraint graph_feedback_scope_requires_target
  check (
    (scope = 'graph')
    or (scope = 'node' and node_id is not null and node_id <> '')
    or (scope = 'edge' and edge_id is not null and edge_id <> '')
  );

-- RLS recommendation:
-- For now, since we insert from a server-side route using the service role,
-- you can keep RLS enabled but without policies, or disable RLS on this table.
-- When you introduce user auth + direct inserts, you'll want explicit policies.
