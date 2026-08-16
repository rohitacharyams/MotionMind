"""semantic_search.py — free-text search over the clip catalog.

Pipeline:
  1. For every clip with metadata, build a "search document" by
     concatenating title + summary + key cues + vibe tags + genre name.
  2. Embed each document with sentence-transformers (MiniLM, CPU,
     ~80 MB model, ~10 ms/embedding).
  3. Cache embeddings to coach/motion_meta_embeddings.npz.
  4. At query time: embed the query, cosine-similarity vs every clip,
     return top-k.

Public API:
    build_index()                  → rebuild from current metadata
    search(query, k=8, genre=None) → list of {id, score, title, summary}

CLI:
    python -m coach.semantic_search build
    python -m coach.semantic_search query "funky slow shoulder roll"
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

COACH = Path(__file__).resolve().parent
META_DIR  = COACH / 'motion_meta'
CACHE_DIR = COACH / 'motion_cache'
EMB_PATH  = COACH / 'motion_meta_embeddings.npz'

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


_model = None
def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _doc_for(meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    if t := meta.get('title'): parts.append(t)
    if s := meta.get('summary'): parts.append(s)
    if tags := meta.get('vibe_tags'): parts.append(' '.join(tags))
    if cues := meta.get('key_cues'):
        parts.append(' '.join(c.get('cue', '') for c in cues
                              if isinstance(c, dict)))
    if mistakes := meta.get('common_mistakes'):
        parts.append(' '.join(mistakes))
    if g := meta.get('genre'): parts.append(g)
    if hint := meta.get('tempo_hint'): parts.append(hint)
    return ' . '.join(p for p in parts if p)


def build_index() -> int:
    model = _get_model()
    ids: List[str] = []
    docs: List[str] = []
    for p in sorted(META_DIR.glob('*.json')):
        try:
            m = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        d = _doc_for(m)
        if not d:
            continue
        ids.append(p.stem)
        docs.append(d)
    if not docs:
        print('[semantic] no metadata found; run `python -m coach.metadata seed` first.')
        return 0
    embs = model.encode(docs, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=32)
    np.savez(EMB_PATH, ids=np.array(ids), embeddings=embs.astype(np.float32))
    print(f'[semantic] indexed {len(ids)} clips → {EMB_PATH}')
    return len(ids)


@lru_cache(maxsize=1)
def _load_index():
    if not EMB_PATH.exists():
        return None
    z = np.load(EMB_PATH, allow_pickle=False)
    return list(z['ids']), z['embeddings']


def _keyword_search(query: str, k: int, genre: Optional[str]
                    ) -> List[Dict[str, Any]]:
    """Fallback when `sentence-transformers` is not installed (prod image
    keeps it out to stay small). Scores each clip's metadata by simple
    bag-of-words overlap with the query."""
    q_terms = {t for t in query.lower().split() if len(t) >= 3}
    if not q_terms:
        return []
    scored: List[Dict[str, Any]] = []
    for p in sorted(META_DIR.glob('*.json')):
        cid = p.stem
        if genre and not cid.startswith(genre):
            continue
        try:
            meta = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        doc = _doc_for(meta).lower()
        score = sum(1 for t in q_terms if t in doc)
        if score == 0:
            continue
        scored.append({
            'id':         cid,
            'score':      float(score) / max(len(q_terms), 1),
            'title':      meta.get('title', ''),
            'summary':    meta.get('summary', ''),
            'difficulty': meta.get('difficulty'),
            'vibe_tags':  meta.get('vibe_tags', []),
        })
    scored.sort(key=lambda r: -r['score'])
    return scored[:k]


def search(query: str, k: int = 8, *, genre: Optional[str] = None
           ) -> List[Dict[str, Any]]:
    # Lazy: if sentence-transformers isn't installed (prod container
    # keeps the image small), or the embeddings index hasn't been built
    # yet, fall back to keyword search over metadata.
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return _keyword_search(query, k, genre)
    idx = _load_index()
    if idx is None:
        return _keyword_search(query, k, genre)
    ids, embs = idx
    model = _get_model()
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = embs @ q
    order = np.argsort(-scores)
    out: List[Dict[str, Any]] = []
    for i in order:
        cid = ids[i]
        if genre and not cid.startswith(genre):
            continue
        meta_path = META_DIR / f'{cid}.json'
        meta = {}
        if meta_path.exists():
            try: meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception: pass
        out.append({
            'id':         cid,
            'score':      float(scores[i]),
            'title':      meta.get('title', ''),
            'summary':    meta.get('summary', ''),
            'difficulty': meta.get('difficulty'),
            'vibe_tags':  meta.get('vibe_tags', []),
        })
        if len(out) >= k:
            break
    return out


if __name__ == '__main__':
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['build', 'query'])
    p.add_argument('text', nargs='*', default=[])
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--genre', default=None)
    a = p.parse_args()
    if a.cmd == 'build':
        build_index()
    elif a.cmd == 'query':
        q = ' '.join(a.text)
        if not q:
            print('usage: query "your text here"', file=sys.stderr); sys.exit(1)
        for r in search(q, k=a.k, genre=a.genre):
            print(f'  {r["score"]:.3f}  {r["id"]}  "{r["title"]}" '
                  f'[{",".join(r["vibe_tags"][:3])}]')
