# Security & Scrub Notes

## Secrets

- **No live API keys are committed.** All credentials are read from environment
  variables and loaded from `.env` files, which are git-ignored. Only
  `*.env.example` placeholder files are tracked.
- Before making this repository public, the maintainer verified there are **no**
  `GROQ`, `Gemini`, `Azure`, storage-account, or other live tokens in tracked files.

> **If you are the original author:** the private source these files came from
> contained real keys in an un-tracked `.env`. Those keys were **never copied**
> here — but you should still **rotate them** (Groq, Gemini/Google AI Studio,
> Azure Speech) since they existed on disk. Rotating is free and takes minutes.

## Reporting a vulnerability

Please open a private security advisory or email the maintainer rather than
filing a public issue for anything sensitive.

## Pre-publish scrub checklist

This repo was extracted from a larger private product. Automated scans confirm no
active third-party trackers or ad IDs remain (Clarity, Amplitude, AdSense, GA4 —
all removed or env-gated to OFF), and all original brand names / domains have been
scrubbed from the code and comments.

- `AUTH_BACKEND_URL` — an **optional** external identity backend. It is **empty by
  default**, so the coach runs fully standalone/anonymous. Set it only if you
  wire up your own `/api/me`-style auth service.
- A few optional integration code paths remain (an auth bridge, a `/api/track`
  proxy). They are inert unless you configure `AUTH_BACKEND_URL`. These are **not
  secrets** — feel free to delete them if you don't need external auth.

To re-scan at any time:

```powershell
# from the repo root — should print nothing security-relevant
Get-ChildItem -Recurse -File -Force -Include *.py,*.js,*.html,*.json,*.md,*.txt |
  Where-Object { $_.FullName -notmatch '\\vendor\\' } |
  Select-String -Pattern 'gsk_[A-Za-z0-9]{15}','AQ\.Ab[A-Za-z0-9]','AccountKey=','xkeysib-','-----BEGIN [A-Z ]*PRIVATE KEY'
```

## Responsible use

- Only extract motion from videos you have the rights to use.
- SMPL/SMPL-X and several motion datasets are **non-commercial / registration
  required** — comply with their licenses (see [ARCHITECTURE.md](ARCHITECTURE.md#assets)).
