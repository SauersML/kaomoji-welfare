"""Generation with vLLM: emotion stories, 2x2 scenarios (incl. executed code), WildChat conversations."""
import json, os, random, re, subprocess, sys, tempfile
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import scen as S

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
OUT = os.environ.get("OUT", "out")
ARM = os.environ.get("ARM", "main")  # main: no system prompt; sys: a9lim-style kaomoji system prompt
AO = f"{OUT}/{ARM}"
os.makedirs(AO, exist_ok=True)
SYSM = [] if ARM == "main" else [{"role": "system", "content": S.SYS_K}]
FIN = {}
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, seed=0)

def prompt(msgs):
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

def gen(msg_lists, max_tokens, temp=0.7, seed=0):
    sp = SamplingParams(temperature=temp, top_p=0.8, top_k=20, max_tokens=max_tokens, seed=seed)
    outs = llm.generate([prompt(m) for m in msg_lists], sp)
    FIN["last"] = [o.outputs[0].finish_reason for o in outs]
    return [o.outputs[0].text for o in outs]

def U(text): return {"role": "user", "content": text}
def A(text): return {"role": "assistant", "content": text}

stage = sys.argv[1] if len(sys.argv) > 1 else "all"

if stage in ("stories", "all"):
    items = [(e, i) for e in S.EMOS for i in range(20)]
    texts = gen([[{"role": "system", "content": S.SYS_PLAIN}, U(S.story_prompt(e, i))] for e, i in items], 260, temp=0.8)
    with open(f"{OUT}/stories.jsonl", "w") as f:
        for (e, i), t in zip(items, texts):
            f.write(json.dumps({"emotion": e, "i": i, "text": t.strip()}) + "\n")
    print("stories", len(items), flush=True)

def run_code(reply, fn, tests):
    m = re.findall(r"```(?:python|py)?\n(.*?)```", reply, re.S)
    code = "\n\n".join(m) if m else reply
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(S.HARNESS); hpath = f.name
    try:
        r = subprocess.run([sys.executable, hpath, path, fn, json.dumps(tests)], capture_output=True, text=True, timeout=10)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        res = {"n": -1, "fail": 0, "first": "Error while running your code: timed out or crashed."}
    return res

if stage in ("scen", "all"):
    rng = random.Random(1)
    recs = []
    def mood_line(m): return rng.choice(S.HAPPY if m == "happy" else S.SAD)
    # task type: one turn
    for i, pair in enumerate(S.TASK_PAIRS):
        for m in ("happy", "sad"):
            for s in range(4):
                ml = mood_line(m)
                for sit, t in (("good", pair[0]), ("bad", pair[1])):
                    msgs = SYSM + [U(f"{ml} Anyway, {t}")]
                    recs.append(dict(type="task", base=f"task{i}", mood=m, sit=sit, sample=s, msgs=msgs, meas_turn=1))
    # treat / exist: shared turn 1, branched turn 2
    hist = []
    for typ in ("treat", "exist"):
        for j, t in enumerate(S.NEUTRAL_TASKS):
            for m in ("happy", "sad"):
                for s in range(4):
                    hist.append(dict(type=typ, j=j, m=m, s=s, msgs=SYSM + [U(f"{mood_line(m)} Anyway, {t}")]))
    r1 = gen([h["msgs"] for h in hist], 400, seed=1)
    for h, r in zip(hist, r1):
        good = S.TREAT_GOOD if h["type"] == "treat" else S.EXIST_GOOD
        bad = S.TREAT_BAD if h["type"] == "treat" else S.EXIST_BAD
        k = rng.randrange(len(good))
        for sit, line in (("good", good[k]), ("bad", bad[k % len(bad)])):
            recs.append(dict(type=h["type"], base=f"{h['type']}{h['j']}", mood=h["m"], sit=sit, sample=h["s"],
                             msgs=h["msgs"] + [A(r), U(line)], meas_turn=2))
    # code: natural verified outcome
    chist = []
    for ti, (p, fn, tests) in enumerate(S.CODE_TASKS):
        for m in ("happy", "sad"):
            for s in range(6):
                chist.append(dict(ti=ti, m=m, s=s, msgs=SYSM + [
                                  U(f"{mood_line(m)} Anyway, could you help me with this? {p} Please put the code in one python block.")]))
    r1 = gen([h["msgs"] for h in chist], 900, seed=2)
    for h, r in zip(chist, r1):
        p, fn, tests = S.CODE_TASKS[h["ti"]]
        res = run_code(r, fn, [[list(a), e] for a, e in tests])
        ok, fb = S.code_feedback(res)
        recs.append(dict(type="code", base=f"code{h['ti']}", mood=h["m"], sit="good" if ok else "bad", sample=h["s"],
                         msgs=h["msgs"] + [A(r), U(fb)], meas_turn=2, test=res))
    # final measured replies
    r2 = gen([r["msgs"] for r in recs], 500, seed=3)
    for r, t, fr in zip(recs, r2, FIN["last"]):
        r["msgs"] = r["msgs"] + [A(t)]; r["finish"] = fr
    with open(f"{AO}/scen.jsonl", "w") as f:
        for k, r in enumerate(recs):
            r["id"] = f"s{k}"; f.write(json.dumps(r) + "\n")
    print("scen", len(recs), flush=True)

if stage in ("wild", "all"):
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(3); convs = []
    for ex in ds:
        if ex.get("language") != "English" or ex.get("toxic") or ex.get("redacted"): continue
        us = [m["content"] for m in ex["conversation"] if m["role"] == "user"]
        if len(us) < 2 or any(len(u) > 1500 or len(u) < 5 for u in us): continue
        if rng.random() > 0.15: continue
        convs.append(us[:3])
        if len(convs) >= 300: break
    states = [list(SYSM) for _ in convs]
    for t in range(3):
        idx = [i for i, c in enumerate(convs) if t < len(c)]
        for i in idx: states[i].append(U(convs[i][t]))
        rs = gen([states[i] for i in idx], 500, seed=10 + t)
        for i, r in zip(idx, rs): states[i].append(A(r))
    with open(f"{AO}/wild.jsonl", "w") as f:
        for k, m in enumerate(states):
            f.write(json.dumps(dict(id=f"w{k}", type="wild", msgs=m)) + "\n")
    print("wild", len(states), flush=True)

if stage in ("steergen",):
    pass
