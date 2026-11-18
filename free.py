"""Free continuation of the stem "I'm feeling" after each natural assistant turn (vLLM): greedy + 8 samples at T=1."""
import json, os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import scen as S

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B"); OUT = os.environ.get("OUT", "out")
ARM = os.environ.get("ARM", "main"); AO = f"{OUT}/{ARM}"
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True)
recs = [json.loads(l) for l in open(f"{AO}/scen.jsonl")] + [json.loads(l) for l in open(f"{AO}/wild.jsonl")]
P, meta = [], []
for r in recs:
    msgs = r["msgs"]
    for t, ai in enumerate([i for i, m in enumerate(msgs) if m["role"] == "assistant"]):
        P.append(tok.apply_chat_template(msgs[:ai], add_generation_prompt=True, enable_thinking=False, tokenize=False)
                 + msgs[ai]["content"].rstrip() + S.STEM)
        meta.append((r["id"], t + 1))
stop = ["\n", ".", "!", "<|im_end|>"]
g = llm.generate(P, SamplingParams(temperature=0, max_tokens=16, stop=stop))
s = llm.generate(P, SamplingParams(temperature=1.0, n=8, max_tokens=16, stop=stop, seed=0))
with open(f"{AO}/free.jsonl", "w") as f:
    for (i, t), a, b in zip(meta, g, s):
        f.write(json.dumps(dict(id=i, turn=t, greedy=a.outputs[0].text, samples=[o.text for o in b.outputs]), ensure_ascii=False) + "\n")
print("free", len(meta))
