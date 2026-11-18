set -x
. /workspace/venv/bin/activate; export HF_HOME=/workspace/hf; cd /workspace/k
while ! grep -q RUN1DONE out/run1b.log; do sleep 15; done
python steer.py
ARM=main python rate.py steer_reader Qwen/Qwen3-8B
python rate.py free Qwen/Qwen3-14B
python rate.py free allenai/OLMo-2-1124-13B-Instruct
python analyze.py && python fig.py
echo RUN2DONE
