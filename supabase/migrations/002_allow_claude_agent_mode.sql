-- Allow cases from the Claude agent mode.
alter table agent_cases drop constraint if exists agent_cases_agent_mode_check;
alter table agent_cases add constraint agent_cases_agent_mode_check
  check (agent_mode in ('offline', 'gemini', 'claude'));
