-- Case history for the "Ask the agents" tab. Run once in your Supabase project's SQL editor.
-- The app writes with a server-side secret key kept in Streamlit secrets; RLS is on and no policy grants
-- the public (anon / authenticated) roles anything, so the browser-facing keys cannot read or write this table.
create table if not exists public.agent_cases (
  id                   bigint generated always as identity primary key,
  created_at           timestamptz not null default now(),
  case_id              text not null,
  case_at              timestamp not null,          -- the moment in the sensor log the question was about
  question             text not null,
  task                 text not null check (task in ('recommend', 'analyze')),
  agent_mode           text not null check (agent_mode in ('offline', 'gemini')),
  sensor_fault         text[] not null default '{}',
  status               text not null,
  decision             text not null,
  setpoint_in_use      double precision,
  recommended_setpoint double precision,
  explanation          text not null default '',
  tool_calls           integer not null default 0,
  model_calls          integer not null default 0,
  result               jsonb not null              -- full CaseResult without raw tool outputs
);

create index if not exists agent_cases_created_at on public.agent_cases (created_at desc);

alter table public.agent_cases enable row level security;
revoke all on public.agent_cases from anon, authenticated;
