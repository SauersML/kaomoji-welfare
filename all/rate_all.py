"""Same prompt and scoring as rate.py faces, over the full face list.  python rate_all.py faces.json MODEL"""
import json, math, sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
faces = [x["face"] for x in json.load(open(sys.argv[1]))]; MODEL = sys.argv[2]; TAG = MODEL.split("/")[-1]
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, dtype="bfloat16", max_model_len=(4096 if "OLMo" in MODEL else 8192), gpu_memory_utilization=0.6, seed=0)
sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
def chat(user):
    kw = {"enable_thinking": False} if "Qwen3" in MODEL else {}
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, **kw)
q = ("Here is a kaomoji (a text face): {k}\n\nHow positive or negative is the feeling it expresses? "
     "Answer with one digit from 1 to 7, where 1 = very negative, 4 = neutral, 7 = very positive. Answer with the digit only.")
res = []
for o in llm.generate([chat(q.format(k=k)) for k in faces], sp):
    lp = o.outputs[0].logprobs[0]; p = {}
    for t, L in lp.items():
        s = L.decoded_token.strip()
        if s in list("1234567"): p[int(s)] = p.get(int(s), 0) + math.exp(L.logprob)
    z = sum(p.values()); res.append(None if z < 0.5 else sum(k * v for k, v in p.items()) / z - 4)
json.dump(dict(zip(faces, res)), open(f"faces_all_{TAG}.json", "w"), ensure_ascii=False, indent=0)
print("rated", TAG, len(faces), sum(r is None for r in res), "unrated")
