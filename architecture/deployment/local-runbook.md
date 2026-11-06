# Local Runbook

[Deployment](README.md)

```bash
.venv/bin/python -m cooling_advisor
```

- Dashboard: http://127.0.0.1:8501
- API health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Stop: Ctrl+C in the launcher terminal.
- Alternative ports: `--api-port 8001 --port 8502`.

The launcher refuses an occupied API port and avoids attaching to an unknown process. It generates a random local token if absent and stores it in `.env` with owner-only permissions. Never print/commit that file. Changing backend code/config requires restarting the launcher; Streamlit development file changes can be rerun through its UI.

To reset no data is deleted automatically. `.local/cooling-advisor.sqlite3` stores cases and review records; `.local/catalog` stores immutable owner studies; `.local/artifacts` preserves imported CSV; `.local/vendor/sustaindc` is the pinned source. Back up these paths before manual lifecycle changes. Lost source catalog files can leave historical case metrics readable but remove timeseries/reproduction context.

If simulator readiness fails, run `.venv/bin/python -m cooling_advisor.simulation install`. Existing changed or unpinned checkouts are not silently reset. If a case was interrupted after reservation, use a new idempotency key; do not reuse the incomplete reservation. If Gemini is unavailable, select offline mode explicitly. If Supabase session expires, sign out and back in.

Run `.venv/bin/python -m pytest -q` for local regression/integration checks. Upstream integration skips explicitly without the pinned source. Configure credentials and run separate provider/Auth/RLS acceptance tests before claiming those integrations work against a live account.
