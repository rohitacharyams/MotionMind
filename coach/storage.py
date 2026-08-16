"""storage.py — pluggable journey/device/push-subscription store.

The coach historically persisted per-user journey docs as JSON files
under ``data/user_journey/<user_id>.json``. That's fine for one box
and a handful of users, but every restart we re-read the disk, and
scaling across more than one App Service instance means file writes
get lost.

This module exposes a single ``JourneyStore`` interface with two
implementations:

* ``JsonJourneyStore`` — original on-disk JSON layout. Always works.
* ``PostgresJourneyStore`` — Azure Postgres Flexible Server backed.
  Auto-creates the schema on first call. Activated when the env var
  ``COACH_DATABASE_URL`` is set (DSN, e.g. ``postgresql://user:pwd@
  host:5432/dbname?sslmode=require``).

The public surface is intentionally tiny and synchronous, matching
the existing call sites in ``server.py``. All Postgres calls use
``psycopg`` (psycopg3) with a single short-lived connection per
operation — journeys are written at session boundaries, not on every
chat tick, so connection overhead is negligible. The storage layer
NEVER raises out to callers: any backend failure logs to stderr and
falls back to the on-disk JSON store so the user-visible experience
keeps working.

Tables created in Postgres
--------------------------
``coach_user_journey``
    user_id (pk text), updated_at timestamptz, doc jsonb
``coach_push_subscriptions``
    user_id text, endpoint text (pk text), platform text,
    subscription jsonb, created_at timestamptz, last_seen_at timestamptz
``coach_device_tokens``
    user_id text, registration_id text (pk text), platform text,
    locale text, tz text, created_at timestamptz, last_seen_at timestamptz
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# psycopg3 is optional — only imported when COACH_DATABASE_URL is set.
try:
    import psycopg  # type: ignore
    from psycopg.types.json import Json  # type: ignore
    _HAS_PSYCOPG = True
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore
    Json = None  # type: ignore
    _HAS_PSYCOPG = False


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _safe_user_key(user_id: str) -> str:
    return ''.join(
        c if (c.isalnum() or c in ('-', '_')) else '_'
        for c in (user_id or '')
    )[:96] or 'anon'


# ─── JSON fallback store ──────────────────────────────────────────────
class JsonJourneyStore:
    """File-backed journey store. The original implementation."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.journey_dir = self.root / 'data' / 'user_journey'
        self.devices_path = self.root / 'data' / 'user_devices.json'
        self.push_path = self.root / 'data' / 'push_subscriptions.json'

    # journey ----------------------------------------------------------
    def _journey_file(self, user_id: str) -> Path:
        return self.journey_dir / f'{_safe_user_key(user_id)}.json'

    def load_journey(self, user_id: Optional[str]) -> Dict[str, Any]:
        if not user_id:
            return {}
        fp = self._journey_file(user_id)
        if not fp.exists():
            return {}
        try:
            raw = json.loads(fp.read_text(encoding='utf-8'))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def save_journey(self, user_id: Optional[str],
                     doc: Dict[str, Any]) -> None:
        if not user_id:
            return
        try:
            self.journey_dir.mkdir(parents=True, exist_ok=True)
            fp = self._journey_file(user_id)
            fp.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                          encoding='utf-8')
        except Exception as e:  # pragma: no cover
            print(f'[storage:json] save_journey failed: {e}', file=sys.stderr)

    def list_journeys(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.journey_dir.exists():
            return out
        for fp in self.journey_dir.glob('*.json'):
            try:
                raw = json.loads(fp.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    raw.setdefault('user_id', fp.stem)
                    out.append(raw)
            except Exception:
                continue
        return out

    # push subscriptions -----------------------------------------------
    def _read_all(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _write_all(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding='utf-8')
        except Exception as e:  # pragma: no cover
            print(f'[storage:json] write {path.name} failed: {e}',
                  file=sys.stderr)

    def save_push_subscription(self, user_id: str,
                               subscription: Dict[str, Any]) -> None:
        if not user_id or not subscription:
            return
        endpoint = subscription.get('endpoint')
        if not endpoint:
            return
        all_subs = self._read_all(self.push_path)
        user_subs = all_subs.setdefault(_safe_user_key(user_id), {})
        user_subs[endpoint] = {
            'subscription': subscription,
            'platform': 'web',
            'created_at': user_subs.get(endpoint, {}).get('created_at')
                          or _utc_iso(),
            'last_seen_at': _utc_iso(),
        }
        self._write_all(self.push_path, all_subs)

    def delete_push_subscription(self, user_id: str, endpoint: str) -> None:
        if not user_id or not endpoint:
            return
        all_subs = self._read_all(self.push_path)
        key = _safe_user_key(user_id)
        if key in all_subs and endpoint in all_subs[key]:
            del all_subs[key][endpoint]
            if not all_subs[key]:
                del all_subs[key]
            self._write_all(self.push_path, all_subs)

    def list_push_subscriptions(self,
                                user_id: Optional[str] = None
                                ) -> List[Dict[str, Any]]:
        all_subs = self._read_all(self.push_path)
        out: List[Dict[str, Any]] = []
        items = (all_subs.get(_safe_user_key(user_id), {}).items()
                 if user_id else
                 [(uid, sub) for uid, subs in all_subs.items()
                              for _, sub in subs.items()])
        if user_id:
            for endpoint, rec in items:
                out.append({'user_id': user_id, 'endpoint': endpoint,
                            **rec})
        else:
            for uid, subs in all_subs.items():
                for endpoint, rec in subs.items():
                    out.append({'user_id': uid, 'endpoint': endpoint,
                                **rec})
        return out

    # native (FCM/APNS via Notification Hubs) tokens -------------------
    def save_device_token(self, user_id: str, registration_id: str,
                          platform: str, locale: str = '',
                          tz: str = '') -> None:
        if not user_id or not registration_id:
            return
        all_devs = self._read_all(self.devices_path)
        user_devs = all_devs.setdefault(_safe_user_key(user_id), {})
        user_devs[registration_id] = {
            'platform': platform or 'unknown',
            'locale': locale,
            'tz': tz,
            'created_at': user_devs.get(registration_id, {}).get('created_at')
                          or _utc_iso(),
            'last_seen_at': _utc_iso(),
        }
        self._write_all(self.devices_path, all_devs)

    def list_device_tokens(self,
                           user_id: Optional[str] = None
                           ) -> List[Dict[str, Any]]:
        all_devs = self._read_all(self.devices_path)
        out: List[Dict[str, Any]] = []
        if user_id:
            for rid, rec in all_devs.get(_safe_user_key(user_id), {}).items():
                out.append({'user_id': user_id,
                            'registration_id': rid, **rec})
        else:
            for uid, devs in all_devs.items():
                for rid, rec in devs.items():
                    out.append({'user_id': uid,
                                'registration_id': rid, **rec})
        return out

    def delete_device_token(self, user_id: str,
                            registration_id: str) -> None:
        if not user_id or not registration_id:
            return
        all_devs = self._read_all(self.devices_path)
        key = _safe_user_key(user_id)
        if key in all_devs and registration_id in all_devs[key]:
            del all_devs[key][registration_id]
            if not all_devs[key]:
                del all_devs[key]
            self._write_all(self.devices_path, all_devs)


# ─── Postgres store ───────────────────────────────────────────────────
class PostgresJourneyStore:
    """Postgres-backed store. Schema is auto-created on first use."""

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS coach_user_journey (
        user_id     text PRIMARY KEY,
        updated_at  timestamptz NOT NULL DEFAULT now(),
        doc         jsonb NOT NULL
    );

    CREATE TABLE IF NOT EXISTS coach_push_subscriptions (
        endpoint     text PRIMARY KEY,
        user_id      text NOT NULL,
        platform     text NOT NULL DEFAULT 'web',
        subscription jsonb NOT NULL,
        created_at   timestamptz NOT NULL DEFAULT now(),
        last_seen_at timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS coach_push_subscriptions_user_idx
        ON coach_push_subscriptions(user_id);

    CREATE TABLE IF NOT EXISTS coach_device_tokens (
        registration_id text PRIMARY KEY,
        user_id         text NOT NULL,
        platform        text NOT NULL,
        locale          text DEFAULT '',
        tz              text DEFAULT '',
        created_at      timestamptz NOT NULL DEFAULT now(),
        last_seen_at    timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS coach_device_tokens_user_idx
        ON coach_device_tokens(user_id);
    """

    def __init__(self, dsn: str, fallback: JsonJourneyStore) -> None:
        self.dsn = dsn
        self.fallback = fallback
        self._schema_ready = False

    # housekeeping -----------------------------------------------------
    def _conn(self):  # type: ignore[no-untyped-def]
        if not _HAS_PSYCOPG:
            raise RuntimeError('psycopg not installed')
        return psycopg.connect(self.dsn, connect_timeout=5)

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        try:
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute(self._SCHEMA_SQL)
            self._schema_ready = True
        except Exception as e:
            print(f'[storage:pg] schema init failed: {e}', file=sys.stderr)
            # Don't raise — fallback path will handle the call.

    # journey ----------------------------------------------------------
    def load_journey(self, user_id: Optional[str]) -> Dict[str, Any]:
        if not user_id:
            return {}
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute(
                        'SELECT doc FROM coach_user_journey WHERE user_id=%s',
                        (user_id,))
                    row = cur.fetchone()
                    if not row:
                        # Lazy one-time migration: an existing user whose
                        # journey still lives only in the legacy JSON store
                        # (e.g. created before Postgres was wired). Read it,
                        # copy it into Postgres, and return it so the user
                        # keeps their memory / streak seamlessly.
                        legacy = self.fallback.load_journey(user_id)
                        if legacy:
                            try:
                                cur.execute(
                                    'INSERT INTO coach_user_journey '
                                    '(user_id, doc, updated_at) '
                                    'VALUES (%s, %s, now()) '
                                    'ON CONFLICT (user_id) DO NOTHING',
                                    (user_id, Json(legacy)))
                            except Exception as _me:           # noqa: BLE001
                                print(f'[storage:pg] lazy-migrate failed: '
                                      f'{_me}', file=sys.stderr)
                        return legacy or {}
                    doc = row[0]
                    return doc if isinstance(doc, dict) else json.loads(doc)
        except Exception as e:
            print(f'[storage:pg] load_journey fallback: {e}', file=sys.stderr)
            return self.fallback.load_journey(user_id)

    def save_journey(self, user_id: Optional[str],
                     doc: Dict[str, Any]) -> None:
        if not user_id:
            return
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO coach_user_journey (user_id, doc, updated_at)
                        VALUES (%s, %s, now())
                        ON CONFLICT (user_id) DO UPDATE
                          SET doc = EXCLUDED.doc,
                              updated_at = now()
                        """,
                        (user_id, Json(doc)))
            # Mirror to JSON so the operator always has a readable copy
            # in source control + recovery is trivial if Postgres dies.
            self.fallback.save_journey(user_id, doc)
        except Exception as e:
            print(f'[storage:pg] save_journey fallback: {e}', file=sys.stderr)
            self.fallback.save_journey(user_id, doc)

    def list_journeys(self) -> List[Dict[str, Any]]:
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute('SELECT user_id, doc FROM coach_user_journey')
                    out: List[Dict[str, Any]] = []
                    for uid, doc in cur.fetchall():
                        d = doc if isinstance(doc, dict) else json.loads(doc)
                        d.setdefault('user_id', uid)
                        out.append(d)
                    return out
        except Exception as e:
            print(f'[storage:pg] list_journeys fallback: {e}', file=sys.stderr)
            return self.fallback.list_journeys()

    # push subscriptions -----------------------------------------------
    def save_push_subscription(self, user_id: str,
                               subscription: Dict[str, Any]) -> None:
        endpoint = (subscription or {}).get('endpoint')
        if not user_id or not endpoint:
            return
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO coach_push_subscriptions
                            (endpoint, user_id, platform, subscription,
                             last_seen_at)
                        VALUES (%s, %s, 'web', %s, now())
                        ON CONFLICT (endpoint) DO UPDATE
                          SET user_id = EXCLUDED.user_id,
                              subscription = EXCLUDED.subscription,
                              last_seen_at = now()
                        """,
                        (endpoint, user_id, Json(subscription)))
        except Exception as e:
            print(f'[storage:pg] save_push fallback: {e}', file=sys.stderr)
            self.fallback.save_push_subscription(user_id, subscription)

    def delete_push_subscription(self, user_id: str, endpoint: str) -> None:
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute(
                        'DELETE FROM coach_push_subscriptions WHERE endpoint=%s',
                        (endpoint,))
        except Exception as e:
            print(f'[storage:pg] delete_push fallback: {e}', file=sys.stderr)
            self.fallback.delete_push_subscription(user_id, endpoint)

    def list_push_subscriptions(self,
                                user_id: Optional[str] = None
                                ) -> List[Dict[str, Any]]:
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    if user_id:
                        cur.execute(
                            'SELECT endpoint, user_id, subscription, '
                            'created_at, last_seen_at '
                            'FROM coach_push_subscriptions WHERE user_id=%s',
                            (user_id,))
                    else:
                        cur.execute(
                            'SELECT endpoint, user_id, subscription, '
                            'created_at, last_seen_at '
                            'FROM coach_push_subscriptions')
                    out = []
                    for endpoint, uid, sub, created, seen in cur.fetchall():
                        s = sub if isinstance(sub, dict) else json.loads(sub)
                        out.append({
                            'endpoint': endpoint,
                            'user_id': uid,
                            'subscription': s,
                            'created_at': created.isoformat()
                                          if created else None,
                            'last_seen_at': seen.isoformat()
                                            if seen else None,
                        })
                    return out
        except Exception as e:
            print(f'[storage:pg] list_push fallback: {e}', file=sys.stderr)
            return self.fallback.list_push_subscriptions(user_id)

    # native tokens ----------------------------------------------------
    def save_device_token(self, user_id: str, registration_id: str,
                          platform: str, locale: str = '',
                          tz: str = '') -> None:
        if not user_id or not registration_id:
            return
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO coach_device_tokens
                            (registration_id, user_id, platform, locale, tz,
                             last_seen_at)
                        VALUES (%s, %s, %s, %s, %s, now())
                        ON CONFLICT (registration_id) DO UPDATE
                          SET user_id = EXCLUDED.user_id,
                              platform = EXCLUDED.platform,
                              locale = EXCLUDED.locale,
                              tz = EXCLUDED.tz,
                              last_seen_at = now()
                        """,
                        (registration_id, user_id, platform or 'unknown',
                         locale or '', tz or ''))
        except Exception as e:
            print(f'[storage:pg] save_device fallback: {e}', file=sys.stderr)
            self.fallback.save_device_token(user_id, registration_id,
                                            platform, locale, tz)

    def list_device_tokens(self,
                           user_id: Optional[str] = None
                           ) -> List[Dict[str, Any]]:
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    if user_id:
                        cur.execute(
                            'SELECT registration_id, user_id, platform, '
                            'locale, tz, created_at, last_seen_at '
                            'FROM coach_device_tokens WHERE user_id=%s',
                            (user_id,))
                    else:
                        cur.execute(
                            'SELECT registration_id, user_id, platform, '
                            'locale, tz, created_at, last_seen_at '
                            'FROM coach_device_tokens')
                    out = []
                    for rid, uid, plat, loc, tz_, created, seen in cur.fetchall():
                        out.append({
                            'registration_id': rid,
                            'user_id': uid,
                            'platform': plat,
                            'locale': loc,
                            'tz': tz_,
                            'created_at': created.isoformat()
                                          if created else None,
                            'last_seen_at': seen.isoformat()
                                            if seen else None,
                        })
                    return out
        except Exception as e:
            print(f'[storage:pg] list_device fallback: {e}', file=sys.stderr)
            return self.fallback.list_device_tokens(user_id)

    def delete_device_token(self, user_id: str,
                            registration_id: str) -> None:
        if not registration_id:
            return
        try:
            self._ensure_schema()
            with self._conn() as cx:
                with cx.cursor() as cur:
                    cur.execute(
                        'DELETE FROM coach_device_tokens '
                        'WHERE registration_id=%s', (registration_id,))
        except Exception as e:
            print(f'[storage:pg] delete_device fallback: {e}',
                  file=sys.stderr)
            self.fallback.delete_device_token(user_id, registration_id)


# ─── module-level singleton ───────────────────────────────────────────
_STORE = None  # type: Optional[Any]


def get_store(root: Path):  # type: ignore[no-untyped-def]
    """Return the active JourneyStore (singleton).

    Picks Postgres when COACH_DATABASE_URL is set AND psycopg is
    importable; otherwise falls back to JSON files. The store is
    constructed once per process.
    """
    global _STORE
    if _STORE is not None:
        return _STORE
    json_store = JsonJourneyStore(root)
    dsn = os.getenv('COACH_DATABASE_URL', '').strip()
    if dsn and _HAS_PSYCOPG:
        try:
            _STORE = PostgresJourneyStore(dsn, fallback=json_store)
            print('[storage] backend=postgres', file=sys.stderr)
            return _STORE
        except Exception as e:
            print(f'[storage] postgres init failed, using json: {e}',
                  file=sys.stderr)
    if dsn and not _HAS_PSYCOPG:
        print('[storage] COACH_DATABASE_URL set but psycopg not '
              'installed; using json. pip install "psycopg[binary]" '
              'to enable Postgres.', file=sys.stderr)
    _STORE = json_store
    print('[storage] backend=json', file=sys.stderr)
    return _STORE


# ─── entry-event log (v179) ───────────────────────────────────────────
# Nothing in this app has ever recorded WHERE a /dance visit came from
# (no document.referrer, no UTM capture, no /api/track-equivalent) — so
# there was no way to see whether the trickle of visitors reaching this
# app is coming from the marketing site, direct links, search, or a
# native wrapper. This is a minimal, best-effort, append-only log of each
# fresh page load's referrer/UTM so that gap becomes measurable.
_ENTRY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS coach_entry_events (
    id           bigserial PRIMARY KEY,
    ts           timestamptz NOT NULL DEFAULT now(),
    referrer     text NOT NULL DEFAULT '',
    utm_source   text NOT NULL DEFAULT '',
    utm_medium   text NOT NULL DEFAULT '',
    utm_campaign text NOT NULL DEFAULT '',
    path         text NOT NULL DEFAULT '',
    user_id      text NOT NULL DEFAULT ''
);
"""
_entry_schema_ready = False


def log_entry_event(root: Path, *, referrer: str = '', utm_source: str = '',
                    utm_medium: str = '', utm_campaign: str = '',
                    path: str = '', user_id: str = '') -> None:
    """Append one entry event. Never raises. Uses Postgres when available
    (durable across restarts / instances), else appends to a local JSONL
    file under ``root / data / entry_events.jsonl``."""
    global _entry_schema_ready
    dsn = os.getenv('COACH_DATABASE_URL', '').strip()
    row = {'referrer': referrer[:2000], 'utm_source': utm_source[:200],
           'utm_medium': utm_medium[:200], 'utm_campaign': utm_campaign[:200],
           'path': path[:500], 'user_id': (user_id or '')[:96]}
    if dsn and _HAS_PSYCOPG:
        try:
            with psycopg.connect(dsn, connect_timeout=5) as cx:
                with cx.cursor() as cur:
                    if not _entry_schema_ready:
                        cur.execute(_ENTRY_SCHEMA_SQL)
                        _entry_schema_ready = True
                    cur.execute(
                        """
                        INSERT INTO coach_entry_events
                            (referrer, utm_source, utm_medium, utm_campaign,
                             path, user_id)
                        VALUES (%(referrer)s, %(utm_source)s, %(utm_medium)s,
                                %(utm_campaign)s, %(path)s, %(user_id)s)
                        """, row)
            return
        except Exception as e:
            print(f'[storage:pg] log_entry_event failed, falling back to '
                  f'json: {e}', file=sys.stderr)
    try:
        fp = Path(root) / 'data' / 'entry_events.jsonl'
        fp.parent.mkdir(parents=True, exist_ok=True)
        row['ts'] = _utc_iso()
        with open(fp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    except Exception as e:  # pragma: no cover
        print(f'[storage:json] log_entry_event failed: {e}', file=sys.stderr)


# ─── skill-event log (v186) — THE RESEARCH MOAT ───────────────────────
# This is the (attempt, instruction, outcome) corpus that no public dataset
# has and that only a DEPLOYED product with real learners moving on camera
# can collect. Each row is one "coaching beat":
#   - an ATTEMPT: the learner tried a move; we captured how well (a live
#     pose-match score 0-100 + which body part was worst + mean error).
#   - the INSTRUCTION the coach gave in response (the cue/correction text,
#     and where it came from: the LLM, the live-feedback heuristic, canned).
#   - the OUTCOME is NOT stored per-row on purpose — it is RECONSTRUCTED
#     offline by ordering rows within one (user_id, clip_id, session_id):
#     "did the score on attempt N+1 improve vs attempt N, given the
#     instruction between them?" That (state=attempt_N, action=instruction,
#     reward=score_delta) triple is exactly the training signal for an
#     offline-RL / contextual-bandit coaching policy. Storing raw beats +
#     computing deltas offline keeps the hot path cheap and keeps every
#     future modelling choice open.
# Append-only, best-effort, never raises. Consented capture only (the app
# must have told the user their movement/coaching data helps improve the
# coach — wire the consent flag through `consent` below).
_SKILL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS coach_skill_events (
    id             bigserial PRIMARY KEY,
    ts             timestamptz NOT NULL DEFAULT now(),
    user_id        text NOT NULL DEFAULT '',
    session_id     text NOT NULL DEFAULT '',
    clip_id        text NOT NULL DEFAULT '',
    event_kind     text NOT NULL DEFAULT '',
    attempt_index  integer NOT NULL DEFAULT 0,
    score          real,
    mean_error     real,
    worst_keypoint text NOT NULL DEFAULT '',
    instruction    text NOT NULL DEFAULT '',
    instruction_source text NOT NULL DEFAULT '',
    consent        boolean NOT NULL DEFAULT false,
    meta           jsonb
);
CREATE INDEX IF NOT EXISTS coach_skill_events_user_clip_idx
    ON coach_skill_events(user_id, clip_id, session_id, ts);
"""
_skill_schema_ready = False


def log_skill_event(root: Path, *, user_id: str = '', session_id: str = '',
                    clip_id: str = '', event_kind: str = '',
                    attempt_index: int = 0, score: Optional[float] = None,
                    mean_error: Optional[float] = None,
                    worst_keypoint: str = '', instruction: str = '',
                    instruction_source: str = '', consent: bool = False,
                    meta: Optional[Dict[str, Any]] = None) -> None:
    """Append one coaching-beat event (attempt score and/or instruction).
    Never raises. Postgres when available (durable, joinable across
    instances), else JSONL under ``root / data / skill_events.jsonl``."""
    global _skill_schema_ready
    dsn = os.getenv('COACH_DATABASE_URL', '').strip()
    row = {
        'user_id': (user_id or '')[:96], 'session_id': (session_id or '')[:96],
        'clip_id': (clip_id or '')[:128], 'event_kind': (event_kind or '')[:32],
        'attempt_index': int(attempt_index or 0),
        'score': (float(score) if score is not None else None),
        'mean_error': (float(mean_error) if mean_error is not None else None),
        'worst_keypoint': (worst_keypoint or '')[:64],
        'instruction': (instruction or '')[:2000],
        'instruction_source': (instruction_source or '')[:32],
        'consent': bool(consent),
        'meta': (meta if isinstance(meta, dict) else None),
    }
    if dsn and _HAS_PSYCOPG:
        try:
            with psycopg.connect(dsn, connect_timeout=5) as cx:
                with cx.cursor() as cur:
                    if not _skill_schema_ready:
                        cur.execute(_SKILL_SCHEMA_SQL)
                        _skill_schema_ready = True
                    cur.execute(
                        """
                        INSERT INTO coach_skill_events
                            (user_id, session_id, clip_id, event_kind,
                             attempt_index, score, mean_error, worst_keypoint,
                             instruction, instruction_source, consent, meta)
                        VALUES (%(user_id)s, %(session_id)s, %(clip_id)s,
                                %(event_kind)s, %(attempt_index)s, %(score)s,
                                %(mean_error)s, %(worst_keypoint)s,
                                %(instruction)s, %(instruction_source)s,
                                %(consent)s, %(meta)s)
                        """,
                        {**row, 'meta': (Json(row['meta'])
                                         if row['meta'] is not None else None)})
            return
        except Exception as e:
            print(f'[storage:pg] log_skill_event failed, falling back to '
                  f'json: {e}', file=sys.stderr)
    try:
        fp = Path(root) / 'data' / 'skill_events.jsonl'
        fp.parent.mkdir(parents=True, exist_ok=True)
        row['ts'] = _utc_iso()
        with open(fp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    except Exception as e:  # pragma: no cover
        print(f'[storage:json] log_skill_event failed: {e}', file=sys.stderr)


# ─── funnel-event log (v202) — WHERE IN THE FIRST 20s DO THEY DROP ─────
# The single biggest unknown in this app's analytics: of everyone who
# reaches /dance, who actually STARTS a guided session vs bounces first,
# and how fast. `coach_user_journey.progress.sessions_completed` only ever
# counts a WS that stayed open >=20s — so a `0` there is indistinguishable
# between "never connected", "connected but bounced in 5s", and "started a
# session then dropped". This append-only funnel log makes each of those a
# distinct, queryable event so the drop-off point is finally measurable:
#   - ws_connected      : the /ws/agent socket opened (t=0 of a visit)
#   - session_started   : the user tapped a length / front-door start
#   - session_completed : a session was recorded (>=20s, streak bumped)
#   - ws_closed         : the socket closed — carries duration_sec + whether
#                         they ever engaged (sent text / started a session)
# Best-effort, never raises. Postgres when available, else JSONL.
_FUNNEL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS coach_funnel_events (
    id           bigserial PRIMARY KEY,
    ts           timestamptz NOT NULL DEFAULT now(),
    event        text NOT NULL DEFAULT '',
    session_id   text NOT NULL DEFAULT '',
    user_id      text NOT NULL DEFAULT '',
    anon         boolean NOT NULL DEFAULT true,
    template_id  text NOT NULL DEFAULT '',
    duration_sec real,
    meta         jsonb
);
CREATE INDEX IF NOT EXISTS coach_funnel_events_ts_idx
    ON coach_funnel_events(ts);
CREATE INDEX IF NOT EXISTS coach_funnel_events_sess_idx
    ON coach_funnel_events(session_id, ts);
"""
_funnel_schema_ready = False


def log_funnel_event(root: Path, *, event: str = '', session_id: str = '',
                     user_id: str = '', anon: bool = True,
                     template_id: str = '', duration_sec: Optional[float] = None,
                     meta: Optional[Dict[str, Any]] = None) -> None:
    """Append one funnel event (ws_connected / session_started /
    session_completed / ws_closed). Never raises. Postgres when available,
    else JSONL under ``root / data / funnel_events.jsonl``."""
    global _funnel_schema_ready
    dsn = os.getenv('COACH_DATABASE_URL', '').strip()
    row = {
        'event': (event or '')[:32], 'session_id': (session_id or '')[:96],
        'user_id': (user_id or '')[:96], 'anon': bool(anon),
        'template_id': (template_id or '')[:64],
        'duration_sec': (float(duration_sec)
                         if duration_sec is not None else None),
        'meta': (meta if isinstance(meta, dict) else None),
    }
    if dsn and _HAS_PSYCOPG:
        try:
            with psycopg.connect(dsn, connect_timeout=5) as cx:
                with cx.cursor() as cur:
                    if not _funnel_schema_ready:
                        cur.execute(_FUNNEL_SCHEMA_SQL)
                        _funnel_schema_ready = True
                    cur.execute(
                        """
                        INSERT INTO coach_funnel_events
                            (event, session_id, user_id, anon, template_id,
                             duration_sec, meta)
                        VALUES (%(event)s, %(session_id)s, %(user_id)s,
                                %(anon)s, %(template_id)s, %(duration_sec)s,
                                %(meta)s)
                        """,
                        {**row, 'meta': (Json(row['meta'])
                                         if row['meta'] is not None else None)})
            return
        except Exception as e:
            print(f'[storage:pg] log_funnel_event failed, falling back to '
                  f'json: {e}', file=sys.stderr)
    try:
        fp = Path(root) / 'data' / 'funnel_events.jsonl'
        fp.parent.mkdir(parents=True, exist_ok=True)
        row['ts'] = _utc_iso()
        with open(fp, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    except Exception as e:  # pragma: no cover
        print(f'[storage:json] log_funnel_event failed: {e}', file=sys.stderr)


def funnel_summary(root: Path, hours: int = 168) -> Dict[str, Any]:
    """Aggregate the funnel over the last ``hours`` into a drop-off report:
    connections, how many started a session, how many completed, and a
    duration-bucket breakdown of ws_closed rows (so 'bounced in <5s without
    starting' is a concrete number). Best-effort; returns {'error': ...} on
    failure. Postgres only (the durable store)."""
    dsn = os.getenv('COACH_DATABASE_URL', '').strip()
    if not (dsn and _HAS_PSYCOPG):
        return {'error': 'no_postgres'}
    try:
        hours = max(1, min(int(hours), 24 * 90))
    except Exception:
        hours = 168
    since_sql = f"now() - interval '{hours} hours'"
    out: Dict[str, Any] = {'window_hours': hours}
    try:
        with psycopg.connect(dsn, connect_timeout=5) as cx:
            with cx.cursor() as cur:
                cur.execute(_FUNNEL_SCHEMA_SQL)
                cur.execute(
                    f"SELECT event, count(*) FROM coach_funnel_events "
                    f"WHERE ts > {since_sql} GROUP BY event")
                counts = {r[0]: int(r[1]) for r in cur.fetchall()}
                out['counts'] = counts
                connected = counts.get('ws_connected', 0)
                started = counts.get('session_started', 0)
                completed = counts.get('session_completed', 0)
                out['funnel'] = {
                    'ws_connected': connected,
                    'session_started': started,
                    'session_completed': completed,
                    'start_rate': (round(started / connected, 3)
                                   if connected else None),
                    'complete_rate': (round(completed / connected, 3)
                                      if connected else None),
                }
                # Duration-bucket breakdown of ws_closed, split by whether
                # they ever started a session.
                cur.execute(
                    f"""
                    SELECT
                      CASE
                        WHEN duration_sec < 5   THEN '0-5s'
                        WHEN duration_sec < 20  THEN '5-20s'
                        WHEN duration_sec < 60  THEN '20-60s'
                        WHEN duration_sec < 300 THEN '1-5m'
                        ELSE '5m+'
                      END AS bucket,
                      coalesce((meta->>'started')::boolean, false) AS started,
                      count(*)
                    FROM coach_funnel_events
                    WHERE event='ws_closed' AND ts > {since_sql}
                    GROUP BY 1, 2 ORDER BY 1, 2
                    """)
                buckets: Dict[str, Dict[str, int]] = {}
                for bucket, was_started, n in cur.fetchall():
                    b = buckets.setdefault(
                        bucket, {'started': 0, 'not_started': 0})
                    b['started' if was_started else 'not_started'] += int(n)
                out['duration_buckets'] = buckets
                # The headline bounce number: closed in <20s having never
                # started a session.
                cur.execute(
                    f"""
                    SELECT count(*) FROM coach_funnel_events
                    WHERE event='ws_closed' AND ts > {since_sql}
                      AND duration_sec < 20
                      AND coalesce((meta->>'started')::boolean, false) = false
                    """)
                out['bounced_under_20s_no_start'] = int(cur.fetchone()[0])
        return out
    except Exception as e:                                        # noqa: BLE001
        return {'error': f'{type(e).__name__}: {e}'}

