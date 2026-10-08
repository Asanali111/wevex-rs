"""Chunk-size benchmark for Wevex recall: BM25, dense vectors, and hybrid (RRF).

Usage: python bench.py <path-to-copy-of-wevex.db> [cache.npz]
Never point this at the live DB; it only reads, but work on a copy anyway.
"""
import sqlite3, ast, re, collections, statistics as st, sys, json, hashlib, os
import numpy as np

DB = sys.argv[1] if len(sys.argv) > 1 else 'wevex_copy.db'
CACHE = sys.argv[2] if len(sys.argv) > 2 else 'emb_cache.npz'
db = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
rows = db.execute("select source_path,line_start,content,rowid from chunks where source_path like '%.py' order by rowid").fetchall()
files = collections.defaultdict(dict)  # path -> {lineno: text}
for p, a, t, _ in rows:  # later rows overwrite earlier (latest version wins)
    for i, ln in enumerate(t.split('\n')):
        files[p][a + i] = ln
src = {}
for p, m in files.items():
    n = max(m)
    src[p] = '\n'.join(m.get(i, '') for i in range(1, n + 1))

# ---- queries from docstrings; docstring text removed from the indexed corpus
STOP = set('the a an of to in is it and or for on with as by be this that are from at if when which its into not no'.split())
queries = []  # (path, def_line, query)
clean = {}
for p, s in src.items():
    try:
        tree = ast.parse(s)
    except SyntaxError:
        continue
    lines = s.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node)
            if not d or not node.body or not isinstance(node.body[0], ast.Expr):
                continue
            first = re.split(r'(?<=[.!?])\s|\n\n', d.strip())[0].replace('\n', ' ')
            ds = node.body[0]
            for i in range(ds.lineno - 1, ds.end_lineno):
                lines[i] = ''
            if len(first.split()) >= 6 and p.startswith('skein/'):
                queries.append((p, node.lineno, first))
    clean[p] = lines

# ---- enclosing-symbol map for preambles
def symbols(p, lo, hi):
    try:
        tree = ast.parse(src[p])
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and lo <= node.lineno <= hi:
            out.append(node.name)
    return out

def modline(p):
    try:
        d = ast.get_docstring(ast.parse(src[p]))
    except SyntaxError:
        d = None
    return (d or '').split('\n')[0][:100]

from fastembed import TextEmbedding
MODEL = TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=os.path.expanduser('~/.cache/fastembed'))
# Separate copy of the model's tokenizer for counting, so budgets are real tokens.
# (Don't touch MODEL's own tokenizer: it must keep truncating at 512.)
import glob
from tokenizers import Tokenizer
TOKENIZER = Tokenizer.from_file(glob.glob(os.path.expanduser(
    '~/.cache/fastembed/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/*/tokenizer.json'))[0])
TOKENIZER.no_truncation()
TOKENIZER.no_padding()
MAX_TOK = 512  # bge-small context; anything longer is silently truncated at embed time

from functools import lru_cache
@lru_cache(maxsize=None)
def tok(t):
    return max(1, len(TOKENIZER.encode(t, add_special_tokens=False).ids))

def chunk(p, budget_tok, overlap=0.15):
    lines = clean[p]
    out, cur, start = [], [], 1
    i = 0
    n = len(lines)
    while i < n:
        j, acc = i, 0
        while j < n and acc + tok(lines[j]) + 1 <= budget_tok:
            acc += tok(lines[j]) + 1
            j += 1
        if j == i:
            j = i + 1
        out.append((i + 1, j, '\n'.join(lines[i:j])))
        if j >= n:
            break
        back = max(1, int((j - i) * overlap))
        i = max(i + 1, j - back)
    return out

def build(cfg):
    chunks = []
    for p in clean:
        if cfg == 'current':
            seen = {}
            for pp, a, t, _ in rows:
                if pp == p:
                    seen[a] = t
            # strip docstrings consistently: use cleaned lines over the same window
            for a in sorted(seen):
                b = a + 79
                chunks.append((p, a, b, '\n'.join(clean[p][a - 1:b])))
            continue
        budget = int(cfg.split('-')[1].split('+')[0])
        for a, b, t in chunk(p, budget):
            if '+pre' in cfg:
                syms = symbols(p, a, b)
                pre = f"File: {p}. Module: {modline(p)}. Defines: {', '.join(syms[:8]) or 'n/a'}.\n"
                t = pre + t
            chunks.append((p, a, b, t))
    return chunks

def words(q):
    ws = [w for w in re.findall(r'[A-Za-z][A-Za-z0-9]+', q.lower()) if w not in STOP]
    return ' OR '.join(ws)

_cache = {}
if os.path.exists(CACHE):
    z = np.load(CACHE)
    _cache = dict(zip(z['k'].tolist(), z['v']))

def embed(texts, query=False):
    keys = [('q:' if query else 'd:') + hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in _cache]
    if todo:
        fn = MODEL.query_embed if query else MODEL.embed
        for (k, _), v in zip(todo, fn([t for _, t in todo], batch_size=64)):
            _cache[k] = v / np.linalg.norm(v)
    return np.stack([_cache[k] for k in keys])

def save_cache():
    np.savez(CACHE, k=np.array(list(_cache.keys())), v=np.stack(list(_cache.values())))

def rank_of(order, ch, p, ln):
    for r, i in enumerate(order[:5]):
        pp, a, b, _ = ch[i]
        if pp == p and a <= ln <= b:
            return r + 1
    return None

def evaluate(cfg):
    ch = build(cfg)
    con = sqlite3.connect(':memory:')
    con.execute("create virtual table c using fts5(t, tokenize='porter unicode61')")
    for i, (p, a, b, t) in enumerate(ch):
        con.execute("insert into c(rowid,t) values(?,?)", (i, t))
    ntok = [tok(t) for *_, t in ch]
    D = embed([t for *_, t in ch])
    Q = embed([q for *_, q in queries], query=True)
    sims = Q @ D.T
    out = {}
    stats = {m: dict(h1=0, h3=0, h5=0, rr=[], t3=[]) for m in ('bm25', 'vec', 'hybrid')}
    for qi, (p, ln, q) in enumerate(queries):
        bm = [r[0] for r in con.execute("select rowid from c where c match ? order by bm25(c) limit 50", (words(q),))]
        vec = list(np.argsort(-sims[qi])[:50])
        rrf = collections.Counter()
        for lst in (bm, vec):
            for r, i in enumerate(lst):
                rrf[i] += 1 / (60 + r)
        hyb = [i for i, _ in rrf.most_common(50)]
        for m, order in (('bm25', bm), ('vec', vec), ('hybrid', hyb)):
            s = stats[m]
            rk = rank_of(order, ch, p, ln)
            s['h1'] += bool(rk and rk <= 1); s['h3'] += bool(rk and rk <= 3); s['h5'] += bool(rk and rk <= 5)
            s['rr'].append(1 / rk if rk else 0)
            s['t3'].append(sum(ntok[i] for i in order[:3]))
    N = len(queries)
    res = dict(cfg=cfg, chunks=len(ch), mean_tok=round(st.mean(ntok)), p90_tok=sorted(ntok)[int(.9 * len(ntok))],
               over_512=round(sum(n > MAX_TOK for n in ntok) / len(ntok), 3))
    for m, s in stats.items():
        res[m] = dict(r1=round(s['h1'] / N, 3), r3=round(s['h3'] / N, 3), r5=round(s['h5'] / N, 3),
                      mrr=round(st.mean(s['rr']), 3), top3_tok=round(st.mean(s['t3'])))
    return res

if __name__ == '__main__':
    print('queries:', len(queries), 'files:', len(clean), file=sys.stderr)
    for cfg in ['current', 'tok-150', 'tok-300', 'tok-400', 'tok-500', 'tok-800',
                'tok-150+pre', 'tok-300+pre', 'tok-400+pre', 'tok-500+pre', 'tok-800+pre']:
        print(json.dumps(evaluate(cfg)), flush=True)
        save_cache()
