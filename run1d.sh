set -x
. /workspace/venv/bin/activate; export HF_HOME=/workspace/hf; cd /workspace/k
ARM=main python rate.py reader Qwen/Qwen3-8B
ARM=sys python rate.py reader Qwen/Qwen3-8B
while [ ! -f out/olmo_ready ]; do sleep 10; done
if grep -q ok out/olmo_ready; then
python rate.py faces allenai/OLMo-2-1124-13B-Instruct
python rate.py free allenai/OLMo-2-1124-13B-Instruct
fi
echo RUN1DONE >> out/run1b.log
echo RUN1DONE
