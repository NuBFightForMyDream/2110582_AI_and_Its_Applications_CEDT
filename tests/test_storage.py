"""Optional Supabase case history: off without keys, never breaks a case, row is JSON-safe."""
import json
from datetime import datetime

from predictive_cooling import storage
from predictive_cooling.agents.models import CaseRequest, CaseResult, Check


class FakeTable:
    def __init__(self, sink, fail=False):
        self.sink, self.fail = sink, fail

    def insert(self, row):
        if self.fail:
            raise ConnectionError("offline")
        self.sink.append(row)
        return self

    def execute(self):
        return self


class FakeClient:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    def table(self, name):
        assert name == storage.TABLE
        return FakeTable(self.rows, self.fail)


def _case():
    at = datetime(2026, 4, 21, 0, 30)
    request = CaseRequest(at=at, sensor_fault=("B6",))
    result = CaseResult(case_id="c1", at=at, mode="recommend", agent_mode="offline", status="completed",
                        decision="change", setpoint_in_use=22, recommended_setpoint=21,
                        evidence={"big": "tool output"}, checks=[Check(name="plan safe", passed=True)])
    return request, result


def test_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert not storage.enabled()
    assert storage.save_case(*_case()) is None
    assert storage.recent_cases() == []


def test_row_is_flat_json_without_evidence():
    row = storage.case_row(*_case())
    json.dumps(row)
    assert row["recommended_setpoint"] == 21 and row["sensor_fault"] == ["B6"]
    assert "evidence" not in row["result"] and row["result"]["checks"][0]["passed"]


def test_save_inserts_and_failures_are_reported_not_raised():
    client = FakeClient()
    assert storage.save_case(*_case(), client=client) is None
    assert client.rows[0]["case_id"] == "c1"
    assert "offline" in storage.save_case(*_case(), client=FakeClient(fail=True))
