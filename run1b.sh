set -x
. /workspace/venv/bin/activate; export HF_HOME=/workspace/hf; cd /workspace/k
python act.py axes
ARM=main python act.py extract
ARM=sys python act.py extract
ARM=main python free.py
ARM=sys python free.py
python rate.py faces Qwen/Qwen3-14B
python rate.py faces allenai/OLMo-2-1124-13B-Instruct
python rate.py free Qwen/Qwen3-14B
python rate.py free allenai/OLMo-2-1124-13B-Instruct
ARM=main python rate.py reader Qwen/Qwen3-8B
ARM=sys python rate.py reader Qwen/Qwen3-8B
echo RUN1DONE
