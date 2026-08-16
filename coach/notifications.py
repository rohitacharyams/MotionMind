"""notifications.py — Web Push + Azure Notification Hubs dispatcher.

This module abstracts pushing a notification to a user across two
transports:

* **Web Push (W3C)** — works on Chrome (Android + desktop), Firefox,
  Edge, and Safari 16+. Requires a VAPID keypair on the server and a
  service worker on the client. No Azure resources required.
* **Azure Notification Hubs** — a single backend "send" call fans out
  to FCM (Android native), APNs (iOS), Baidu, WNS, etc. Required when
  shipping the Android Play Store wrapper since the wrapped WebView
  does NOT receive Web Push when the app is fully backgrounded.

Both transports degrade gracefully when their environment variables
are missing — calls become no-ops and ``send_to_user`` returns counts
of {sent, failed, skipped} per channel. That means dev can run with
zero config and prod can flip on either channel by setting env vars.

ENV
---
``VAPID_PUBLIC_KEY``, ``VAPID_PRIVATE_KEY``  (Web Push)
    Base64url-encoded P-256 keys. Use
    ``python -m coach.notifications keygen`` to generate a pair.
``VAPID_CLAIM_EMAIL``
    Contact email injected as the JWT ``sub`` (default
    ``mailto:hello@example.com``).
``AZURE_NOTIFICATION_HUB_CONNECTION``
    Connection string (DefaultFullSharedAccessSignature) for the hub.
``AZURE_NOTIFICATION_HUB_NAME``
    The hub name (NOT the namespace).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Optional: pywebpush handles the encryption + VAPID JWT for us.
try:
    from pywebpush import webpush, WebPushException  # type: ignore
    _HAS_WEBPUSH = True
except Exception:
    webpush = None  # type: ignore
    WebPushException = Exception  # type: ignore
    _HAS_WEBPUSH = False

try:
    from py_vapid import Vapid  # type: ignore
    _HAS_VAPID = True
except Exception:
    Vapid = None  # type: ignore
    _HAS_VAPID = False


# ─── module-level config (cached at import) ─────────────────────────
VAPID_PUBLIC_KEY = os.getenv('VAPID_PUBLIC_KEY', '').strip()
VAPID_PRIVATE_KEY = os.getenv('VAPID_PRIVATE_KEY', '').strip()
VAPID_EMAIL = os.getenv('VAPID_CLAIM_EMAIL',
                        'mailto:hello@example.com').strip()

HUB_CONN = os.getenv('AZURE_NOTIFICATION_HUB_CONNECTION', '').strip()
HUB_NAME = os.getenv('AZURE_NOTIFICATION_HUB_NAME', '').strip()

# Direct FCM HTTP v1: the full Firebase *service account* JSON (Project
# settings -> Service accounts -> Generate new private key) pasted as a
# single app-setting string. This is what lets us send native Android
# push without Azure Notification Hubs. Legacy FCM server keys were
# retired by Google in 2024, so v1 (OAuth2 + service account) is the
# only supported path.
FCM_SA_JSON = os.getenv('FCM_SERVICE_ACCOUNT_JSON', '').strip()


@dataclass
class SendResult:
    """Per-channel send tally."""
    web_sent: int = 0
    web_failed: int = 0
    hub_sent: int = 0
    hub_failed: int = 0
    fcm_sent: int = 0
    fcm_failed: int = 0
    skipped: int = 0
    errors: List[str] = None  # type: ignore
    # FCM device tokens that came back dead (UNREGISTERED / not found) so
    # the caller can prune them from storage.
    stale_tokens: List[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.stale_tokens is None:
            self.stale_tokens = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            'web_sent': self.web_sent, 'web_failed': self.web_failed,
            'hub_sent': self.hub_sent, 'hub_failed': self.hub_failed,
            'fcm_sent': self.fcm_sent, 'fcm_failed': self.fcm_failed,
            'skipped': self.skipped, 'errors': self.errors[-5:],
        }


# ─── Web Push ────────────────────────────────────────────────────────
def web_push_enabled() -> bool:
    return bool(VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY and _HAS_WEBPUSH)


def public_vapid_key() -> str:
    """Return the public VAPID key for the browser to subscribe with."""
    return VAPID_PUBLIC_KEY


def generate_vapid_keypair() -> Tuple[str, str]:
    """Generate a new VAPID P-256 keypair, base64url-encoded.

    Returns ``(public_key_b64url, private_key_b64url)``.

    The public key is what you set in ``VAPID_PUBLIC_KEY`` and pass
    to the browser; the private key goes in ``VAPID_PRIVATE_KEY``.
    """
    if not _HAS_VAPID:
        raise RuntimeError(
            'py-vapid not installed. Run: pip install py-vapid '
            'cryptography pywebpush')
    v = Vapid()
    v.generate_keys()
    # py-vapid exposes raw EC keys; we want url-safe base64.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    priv = v.private_key
    pub = priv.public_key()
    raw_priv = priv.private_numbers().private_value.to_bytes(32, 'big')
    raw_pub = pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint)
    def b64u(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b'=').decode('ascii')
    return b64u(raw_pub), b64u(raw_priv)


def _send_web_push(subscription: Dict[str, Any],
                   payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Send one Web Push. Returns (ok, error_message_or_empty)."""
    if not web_push_enabled():
        return False, 'web_push_disabled'
    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload, ensure_ascii=False),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={'sub': VAPID_EMAIL},
            ttl=12 * 3600,
        )
        return True, ''
    except WebPushException as e:  # type: ignore[misc]
        status = getattr(getattr(e, 'response', None), 'status_code', 0)
        return False, f'web_push:{status}:{e}'
    except Exception as e:
        return False, f'web_push:exc:{e}'


# ─── Azure Notification Hubs (REST) ──────────────────────────────────
def _parse_hub_conn(conn: str) -> Tuple[str, str, str]:
    """Parse the DefaultFullSharedAccessSignature connection string.

    Returns (endpoint, key_name, key_value).
    """
    endpoint, key_name, key_value = '', '', ''
    for chunk in conn.split(';'):
        if not chunk:
            continue
        if '=' not in chunk:
            continue
        k, v = chunk.split('=', 1)
        if k == 'Endpoint':
            endpoint = v.replace('sb://', 'https://').rstrip('/')
        elif k == 'SharedAccessKeyName':
            key_name = v
        elif k == 'SharedAccessKey':
            key_value = v
    return endpoint, key_name, key_value


def _hub_sas(uri: str, key_name: str, key_value: str,
             ttl_sec: int = 3600) -> str:
    """Generate a SAS token for one Notification Hubs REST call."""
    target = urllib.parse.quote(uri, safe='').lower()
    expiry = int(time.time() + ttl_sec)
    string_to_sign = f'{target}\n{expiry}'
    sig = base64.b64encode(hmac.new(
        key_value.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        hashlib.sha256).digest())
    return (f'SharedAccessSignature sr={target}'
            f'&sig={urllib.parse.quote(sig)}'
            f'&se={expiry}&skn={key_name}')


def hub_enabled() -> bool:
    return bool(HUB_CONN and HUB_NAME)


def _send_hub_notification(payload: Dict[str, Any],
                           tag: Optional[str] = None) -> Tuple[bool, str]:
    """Send a single notification via Notification Hubs (Direct send).

    Payload shape:
        {'fcm': {...}, 'apns': {...}}  → routed by Hub per platform.
    """
    if not hub_enabled():
        return False, 'hub_disabled'
    endpoint, key_name, key_value = _parse_hub_conn(HUB_CONN)
    if not all((endpoint, key_name, key_value)):
        return False, 'hub_bad_conn'
    url = (f'{endpoint}/{HUB_NAME}/messages/'
           '?api-version=2015-01')
    sas = _hub_sas(url, key_name, key_value)
    headers = {
        'Authorization': sas,
        'Content-Type': 'application/json;charset=utf-8',
        # 'template' format means Notification Hubs picks the right
        # native payload for each registered device.
        'ServiceBusNotification-Format': 'template',
    }
    if tag:
        headers['ServiceBusNotification-Tags'] = tag
    body = json.dumps(payload, ensure_ascii=False)
    try:
        with httpx.Client(timeout=8.0) as cx:
            r = cx.post(url, headers=headers, content=body)
        if 200 <= r.status_code < 300:
            return True, ''
        return False, f'hub:{r.status_code}:{r.text[:200]}'
    except Exception as e:
        return False, f'hub:exc:{e}'


# ─── FCM HTTP v1 (direct, no Notification Hubs) ─────────────────────
_fcm_sa_cache: Optional[Dict[str, Any]] = None
# {'access_token': str, 'exp': int-epoch}
_fcm_token_cache: Dict[str, Any] = {'access_token': '', 'exp': 0}


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b'=').decode('ascii')


def _fcm_service_account() -> Optional[Dict[str, Any]]:
    """Parse + cache the service-account JSON. Returns None when unset
    or malformed (so every FCM call degrades to a no-op)."""
    global _fcm_sa_cache
    if _fcm_sa_cache is not None:
        return _fcm_sa_cache or None
    if not FCM_SA_JSON:
        _fcm_sa_cache = {}
        return None
    try:
        sa = json.loads(FCM_SA_JSON)
        if sa.get('private_key') and sa.get('client_email') \
                and sa.get('project_id'):
            _fcm_sa_cache = sa
            return sa
        print('[fcm] service-account JSON missing required fields',
              file=sys.stderr)
    except Exception as e:
        print(f'[fcm] bad FCM_SERVICE_ACCOUNT_JSON: {e}', file=sys.stderr)
    _fcm_sa_cache = {}
    return None


def fcm_v1_enabled() -> bool:
    return _fcm_service_account() is not None


def _fcm_access_token() -> Tuple[str, str]:
    """Mint (and cache ~1h) a Google OAuth2 access token for the FCM
    scope by signing a JWT with the service-account private key.
    Returns (access_token, error)."""
    sa = _fcm_service_account()
    if not sa:
        return '', 'fcm_disabled'
    now = int(time.time())
    if _fcm_token_cache['access_token'] and _fcm_token_cache['exp'] - 60 > now:
        return _fcm_token_cache['access_token'], ''
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        header = _b64u(json.dumps({'alg': 'RS256', 'typ': 'JWT'}).encode())
        claims = _b64u(json.dumps({
            'iss': sa['client_email'],
            'scope': 'https://www.googleapis.com/auth/firebase.messaging',
            'aud': 'https://oauth2.googleapis.com/token',
            'iat': now, 'exp': now + 3600,
        }).encode())
        signing_input = f'{header}.{claims}'.encode()
        key = serialization.load_pem_private_key(
            sa['private_key'].encode('utf-8'), password=None)
        sig = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        assertion = f'{header}.{claims}.{_b64u(sig)}'
        with httpx.Client(timeout=8.0) as cx:
            r = cx.post('https://oauth2.googleapis.com/token', data={
                'grant_type':
                    'urn:ietf:params:oauth:grant-type:jwt-bearer',
                'assertion': assertion,
            })
        if r.status_code != 200:
            return '', f'fcm_token:{r.status_code}:{r.text[:180]}'
        tok = r.json()
        _fcm_token_cache['access_token'] = tok.get('access_token', '')
        _fcm_token_cache['exp'] = now + int(tok.get('expires_in', 3600))
        return _fcm_token_cache['access_token'], ''
    except Exception as e:
        return '', f'fcm_token:exc:{e}'


def _send_fcm_v1(token: str,
                 payload: Dict[str, Any]) -> Tuple[bool, str, bool]:
    """Send one native push via FCM HTTP v1.

    Returns ``(ok, error, is_stale)``. ``is_stale`` is True when the
    device token is dead (app uninstalled / token rotated) so the caller
    can delete it from storage.
    """
    sa = _fcm_service_account()
    if not sa:
        return False, 'fcm_disabled', False
    access, err = _fcm_access_token()
    if not access:
        return False, err, False
    url = (f'https://fcm.googleapis.com/v1/projects/'
           f'{sa["project_id"]}/messages:send')
    message = {
        'message': {
            'token': token,
            'notification': {
                'title': payload.get('title', 'Dance.AI'),
                'body': payload.get('body', ''),
            },
            'data': {'url': str(payload.get('url', '/'))},
            'android': {
                'priority': 'high',
                'notification': {
                    'default_sound': True,
                    'notification_priority': 'PRIORITY_HIGH',
                },
            },
        }
    }
    try:
        with httpx.Client(timeout=8.0) as cx:
            r = cx.post(url, headers={
                'Authorization': f'Bearer {access}',
                'Content-Type': 'application/json',
            }, content=json.dumps(message, ensure_ascii=False))
        if 200 <= r.status_code < 300:
            return True, '', False
        text = r.text or ''
        # 404 or UNREGISTERED => token is permanently dead; prune it.
        stale = (r.status_code == 404 or 'UNREGISTERED' in text
                 or 'registration-token-not-registered' in text)
        return False, f'fcm:{r.status_code}:{text[:180]}', stale
    except Exception as e:
        return False, f'fcm:exc:{e}', False


# ─── high-level dispatcher ──────────────────────────────────────────
def send_to_user(user_id: str, title: str, body: str,
                 url: str = '/', subscriptions: Optional[List[Dict]] = None,
                 hub_tag: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None,
                 device_tokens: Optional[List[str]] = None) -> SendResult:
    """Push a notification to one user across all configured channels.

    ``subscriptions`` is the list of Web Push subscription dicts for
    that user (as stored by :mod:`coach.storage`). When omitted the
    caller is responsible for handling Web Push elsewhere; only Azure
    Hubs will be attempted.

    The Notification Hubs call is fired with ``hub_tag`` (typically
    ``f'user:{user_id}'``) so a single REST call lights up all of
    that user's registered native devices.
    """
    result = SendResult()
    payload = {
        'title': title or 'Dance.AI',
        'body': body or '',
        'url': url or '/',
        'ts': int(time.time()),
    }
    if extra:
        payload.update(extra)

    # --- Web Push (per subscription) ---
    if web_push_enabled() and subscriptions:
        for sub in subscriptions:
            ok, err = _send_web_push(sub, payload)
            if ok:
                result.web_sent += 1
            else:
                result.web_failed += 1
                result.errors.append(err)
    elif subscriptions:
        result.skipped += len(subscriptions)

    # --- Azure Notification Hubs (one tag-targeted call) ---
    if hub_enabled():
        hub_payload = {
            # 'template' body uses simple key tokens. The hub turns
            # this into the right FCM / APNs payload per device.
            'title': payload['title'],
            'body': payload['body'],
            'url': payload['url'],
        }
        ok, err = _send_hub_notification(hub_payload,
                                          tag=hub_tag or f'user:{user_id}')
        if ok:
            result.hub_sent += 1
        else:
            result.hub_failed += 1
            result.errors.append(err)

    # --- FCM HTTP v1 (direct, per device token) ---
    if fcm_v1_enabled() and device_tokens:
        for tok in device_tokens:
            if not tok:
                continue
            ok, err, stale = _send_fcm_v1(tok, payload)
            if ok:
                result.fcm_sent += 1
            else:
                result.fcm_failed += 1
                result.errors.append(err)
                if stale:
                    result.stale_tokens.append(tok)
    elif device_tokens:
        result.skipped += len(device_tokens)
    return result


def diagnostics() -> Dict[str, Any]:
    """Snapshot of which channels are live (for /api/notifications/diag)."""
    sa = _fcm_service_account()
    return {
        'web_push_enabled': web_push_enabled(),
        'web_push_libs': {
            'pywebpush': _HAS_WEBPUSH,
            'py_vapid': _HAS_VAPID,
        },
        'vapid_public_set': bool(VAPID_PUBLIC_KEY),
        'hub_enabled': hub_enabled(),
        'hub_name': HUB_NAME or None,
        'fcm_v1_enabled': fcm_v1_enabled(),
        'fcm_project': (sa or {}).get('project_id'),
    }


# ─── CLI entry: keygen ──────────────────────────────────────────────
def _cli() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print('usage: python -m coach.notifications keygen')
        return 0
    if args[0] == 'keygen':
        try:
            pub, priv = generate_vapid_keypair()
        except Exception as e:
            print(f'keygen failed: {e}', file=sys.stderr)
            return 1
        print('VAPID_PUBLIC_KEY=' + pub)
        print('VAPID_PRIVATE_KEY=' + priv)
        print('# add both to your .env then restart the server')
        return 0
    if args[0] == 'diag':
        print(json.dumps(diagnostics(), indent=2))
        return 0
    print(f'unknown command: {args[0]}', file=sys.stderr)
    return 2


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(_cli())
