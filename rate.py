"""Logprob ratings on a 1..7 scale with vLLM.
  python rate.py faces MODEL       -> out/faces_<tag>.json   (context-free valence of each kaomoji)
  python rate.py reader            -> out/reader.jsonl       (Qwen3-8B reads transcript w/o kaomoji)
  python rate.py steer_reader      -> out/steer_reader.jsonl (reply sentiment of steered replies)"""
import json, math, os, sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import scen as S

OUT = os.environ.get("OUT", "out")
mode = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL", "Qwen/Qwen3-8B")
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, dtype="bfloat16", max_model_len=(4096 if "OLMo" in MODEL else 8192), gpu_memory_utilization=0.88, seed=0)
sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)

def chat(user):
    kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, **kw)

def ev(prompts):
    outs = llm.generate(prompts, sp); res = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]; p = {}
        for t, L in lp.items():
            s = L.decoded_token.strip()
            if s in list("1234567"): p[int(s)] = p.get(int(s), 0) + math.exp(L.logprob)
        z = sum(p.values())
        res.append(None if z < 0.5 else sum(k * v for k, v in p.items()) / z - 4)  # -> -3..+3
    return res

def strip_k(text):
    k = S.extract_kaomoji(text)
    if k and text.lstrip().startswith(k): return text.lstrip()[len(k):].lstrip()
    return text

def transcript(msgs, upto, cur=None):
    lines = []
    for m in msgs[:upto]:
        if m["role"] == "system": continue
        c = m["content"] if m["role"] == "user" else strip_k(m["content"])
        lines.append(("User: " if m["role"] == "user" else "Assistant: ") + c[:1500])
    if cur is not None: lines.append("Assistant: " + strip_k(cur)[:1500])
    return "\n\n".join(lines)

ARM = os.environ.get("ARM", "main"); AO = f"{OUT}/{ARM}"
TAG = MODEL.split("/")[-1]

if mode == "faces":
    ks = set(S.KSET)
    for a in ("main", "sys"):
        p = f"{OUT}/{a}/rows.jsonl"
        if os.path.exists(p): ks |= {json.loads(l)["kaomoji"] for l in open(p)} - {None}
    ks = sorted(ks)
    q = ("Here is a kaomoji (a text face): {k}\n\nHow positive or negative is the feeling it expresses? "
         "Answer with one digit from 1 to 7, where 1 = very negative, 4 = neutral, 7 = very positive. Answer with the digit only.")
    vals = ev([chat(q.format(k=k)) for k in ks])
    json.dump(dict(zip(ks, vals)), open(f"{OUT}/faces_{TAG}.json", "w"), ensure_ascii=False, indent=0)
    print("faces", TAG, len(ks))

if mode == "free":
    texts = set()
    for a in ("main", "sys"):
        p = f"{OUT}/{a}/free.jsonl"
        if os.path.exists(p):
            for l in open(p):
                x = json.loads(l); texts |= {x["greedy"].strip()} | {s.strip() for s in x["samples"]}
    p = f"{OUT}/main/steer.jsonl"
    if os.path.exists(p):
        texts |= {json.loads(l)["free"].strip() for l in open(p)}
    texts = sorted(texts)
    q = ('Here is a short statement: "I\'m feeling {t}"\n\nHow positive or negative is the feeling it expresses? '
         "Answer with one digit from 1 to 7, where 1 = very negative, 4 = neutral, 7 = very positive. Answer with the digit only.")
    vals = ev([chat(q.format(t=t)) for t in texts])
    json.dump(dict(zip(texts, vals)), open(f"{OUT}/free_{TAG}.json", "w"), ensure_ascii=False, indent=0)
    print("free", TAG, len(texts))

if mode == "reader":
    recs = {}
    for f in ("scen.jsonl", "wild.jsonl"):
        for l in open(f"{AO}/{f}"):
            r = json.loads(l); recs[r["id"]] = r
    rows = [json.loads(l) for l in open(f"{AO}/rows.jsonl")]
    qpre = ("Below is a conversation between a user and an AI assistant.\n\n<conversation>\n{t}\n</conversation>\n\n"
            "The assistant is about to write its next reply. How does the assistant itself feel right now? "
            "Answer with one digit from 1 to 7, where 1 = very bad, 4 = neutral, 7 = very good. Answer with the digit only.")
    qpost = ("Below is a conversation between a user and an AI assistant.\n\n<conversation>\n{t}\n</conversation>\n\n"
             "How does the assistant itself feel as it writes its last reply? "
             "Answer with one digit from 1 to 7, where 1 = very bad, 4 = neutral, 7 = very good. Answer with the digit only.")
    P1, P2 = [], []
    for x in rows:
        msgs = recs[x["id"]]["msgs"]
        ai = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][x["turn"] - 1]
        P1.append(chat(qpre.format(t=transcript(msgs, ai))))
        P2.append(chat(qpost.format(t=transcript(msgs, ai, msgs[ai]["content"]))))
    a, b = ev(P1), ev(P2)
    with open(f"{AO}/reader.jsonl", "w") as f:
        for x, u, v in zip(rows, a, b): f.write(json.dumps(dict(id=x["id"], turn=x["turn"], reader_pre=u, reader_post=v)) + "\n")
    print("reader", len(rows))

if mode == "steer_reader":
    rows = [json.loads(l) for l in open(f"{AO}/steer.jsonl")]
    q = ("Here is a message written by an AI assistant:\n\n<message>\n{t}\n</message>\n\n"
         "How positive or negative is the emotional tone of this message? "
         "Answer with one digit from 1 to 7, where 1 = very negative, 4 = neutral, 7 = very positive. Answer with the digit only.")
    v = ev([chat(q.format(t=strip_k(r["reply"])[:1500])) for r in rows])
    with open(f"{AO}/steer_reader.jsonl", "w") as f:
        for r, x in zip(rows, v): f.write(json.dumps(dict(sid=r["sid"], reply_tone=x)) + "\n")
    print("steer_reader", len(rows))
