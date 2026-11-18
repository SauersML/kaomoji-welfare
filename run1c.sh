set -x
. /workspace/venv/bin/activate; export HF_HOME=/workspace/hf; cd /workspace/k
export HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-14B'); print('dl14')"
python -c "from huggingface_hub import snapshot_download as s; s('allenai/OLMo-2-1124-13B-Instruct'); print('dlolmo')" &
python rate.py faces Qwen/Qwen3-14B
python rate.py free Qwen/Qwen3-14B
wait
python rate.py faces allenai/OLMo-2-1124-13B-Instruct
python rate.py free allenai/OLMo-2-1124-13B-Instruct
ARM=main python rate.py reader Qwen/Qwen3-8B
ARM=sys python rate.py reader Qwen/Qwen3-8B
echo RUN1DONE >> out/run1b.log
echo RUN1DONE
