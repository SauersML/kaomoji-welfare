"""Collect kaomoji from three public dictionaries, filter, (optionally) dedupe by Qwen3-8B tokens of ' '+face."""
import json, re, sys, unicodedata, os
import pyarrow as pa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scen import KSET
SRC = os.environ.get("SRC", "src")
cand = {}  # face -> set(sources)
def add(f, s):
    if not isinstance(f, str): return
    f = unicodedata.normalize("NFC", f).strip()
    if f: cand.setdefault(f, set()).add(s)
for k in json.load(open(f"{SRC}/emoticon_kaomoji_dataset/emoticon_dict.json")): add(k, "ekohrt")
for x in json.load(open(f"{SRC}/kaomoji-data/kaomoji.json")): add(x["text"], "kaomojikan")
try:
    t = pa.ipc.open_stream(open(f"{SRC}/mrzjy.arrow", "rb")).read_all()
except Exception:
    t = pa.ipc.open_file(open(f"{SRC}/mrzjy.arrow", "rb")).read_all()
col = [c for c in t.column_names if "kaomoji" in c.lower() or c.lower() in ("text", "emoticon")]
for v in t.column(col[0]).to_pylist(): add(v, "mrzjy")
n_raw = len(cand)
WORD = re.compile(r"[A-Za-z]{2,}")
PICTO = re.compile(r"[\U0001F000-\U0001FAFF\u200d\ufe0f\U000E0000-\U000E007F]")
OPEN = set("(（ʕ༼[［{｛【〔")
def why(f):
    if "\n" in f or "\t" in f: return "multiline"
    if WORD.search(f): return "letters"
    if PICTO.search(f): return "emoji"
    if not any(c in OPEN for c in f) and not any(c in f for c in "ツ"): return "no_bracket"
    if sum(f.count(c) for c in "(（ʕ༼") > 2: return "multi_face"
    if len(f) < 3: return "short"
    if any(unicodedata.category(c) in ("Cc", "Co", "Cn") for c in f): return "ctrl"
    return None
reasons = {}; keep = []
for f in cand:
    r = why(f)
    if f in KSET: r = None
    reasons[r] = reasons.get(r, 0) + 1
    if r is None: keep.append(f)
print("raw", n_raw, "by source", {s: sum(s in v for v in cand.values()) for s in ("ekohrt", "kaomojikan", "mrzjy")}, "filter", reasons)
if len(sys.argv) > 1:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    out = {}; drop_len = 0
    order = list(KSET) + sorted(f for f in keep if f not in KSET)
    for f in order:
        ids = tuple(tok(" " + f, add_special_tokens=False)["input_ids"])
        if len(ids) > 20 and f not in KSET: drop_len += 1; continue
        if ids in out: continue
        out[ids] = f
    faces = [dict(face=f, ntok=len(ids), sources=sorted(cand[f]) if f in cand else ["KSET"], orig45=f in KSET) for ids, f in out.items()]
    assert all(k in [x["face"] for x in faces] for k in KSET)
    json.dump(faces, open(sys.argv[1], "w"), ensure_ascii=False, indent=0)
    print("kept", len(keep), "dropped >20 tok", drop_len, "after token dedupe", len(faces))
