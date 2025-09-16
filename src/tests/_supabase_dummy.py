from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class _Record:
    op: str
    table: str
    payload: Dict[str, Any]
    filters: List[Tuple[str, Any]] = field(default_factory=list)


class DummySupabase:
    """Petit double Supabase pour les tests unitaires."""

    def __init__(self) -> None:
        self.records: list[_Record] = []
        self.select_responses: dict[str, list[list[dict[str, Any]]]] = {}
        self.next_ids: dict[str, int] = {"jobs": 1, "files": 1, "videos": 1}
        self.auth = type("Auth", (), {})()

    def queue_select(self, table: str, rows: list[dict[str, Any]]) -> None:
        self.select_responses.setdefault(table, []).append(rows)

    def table(self, name: str):
        return _DummyTable(self, name)


class _DummyTable:
    def __init__(self, supabase: DummySupabase, name: str) -> None:
        self.supabase = supabase
        self.name = name
        self._filters: list[tuple[str, Any]] = []

    # Chaîne de méthodes utilisée par supabase-py
    def select(self, *args: Any, **kwargs: Any):  # pragma: no cover - arguments ignorés
        return self

    def insert(self, payload: dict[str, Any]):
        return _DummyInsert(self.supabase, self.name, payload)

    def update(self, payload: dict[str, Any]):
        return _DummyUpdate(self.supabase, self.name, payload)

    def eq(self, column: str, value: Any):
        self._filters.append(("eq", column, value))
        return self

    def in_(self, column: str, values: list[Any]):
        self._filters.append(("in", column, tuple(values)))
        return self

    def order(self, column: str, desc: bool = False):  # pragma: no cover
        self._filters.append(("order", column, desc))
        return self

    def limit(self, count: int):  # pragma: no cover - stocké pour debug
        self._filters.append(("limit", count))
        return self

    @property
    def not_(self):
        return _DummyNot(self)

    def execute(self):
        rows = self.supabase.select_responses.get(self.name, [])
        data = rows.pop(0) if rows else []
        return type("Resp", (), {"data": data})()


class _DummyInsert:
    def __init__(self, supabase: DummySupabase, table: str, payload: dict[str, Any]) -> None:
        self.supabase = supabase
        self.table = table
        self.payload = payload

    def execute(self):
        record = dict(self.payload)
        if self.table in self.supabase.next_ids and "id" not in record:
            next_id = self.supabase.next_ids[self.table]
            record["id"] = next_id
            self.supabase.next_ids[self.table] = next_id + 1
        self.supabase.records.append(_Record("insert", self.table, record))
        return type("Resp", (), {"data": [record]})()


class _DummyUpdate:
    def __init__(self, supabase: DummySupabase, table: str, payload: dict[str, Any]) -> None:
        self.supabase = supabase
        self.table = table
        self.payload = payload
        self.filters: list[tuple[str, Any]] = []

    def eq(self, column: str, value: Any):
        self.filters.append(("eq", column, value))
        return self

    def execute(self):
        self.supabase.records.append(
            _Record("update", self.table, self.payload, list(self.filters))
        )
        return type("Resp", (), {"data": []})()


class _DummyNot:
    def __init__(self, table: _DummyTable) -> None:
        self.table = table

    def is_(self, column: str, value: Any):  # pragma: no cover - simple enchainement
        self.table._filters.append(("not.is", column, value))
        return self.table
