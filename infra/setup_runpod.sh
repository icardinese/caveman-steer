#!/usr/bin/env bash
# Run once per fresh Colab runtime, AFTER mounting Drive from a notebook cell:
#     from google.colab import drive; drive.mount('/content/drive')
# This script can't do that mount itself -- the auth handshake needs the Colab frontend, not a
# subprocess. Everything below just assumes the Shared Drive already exists and errors loudly
# if it doesn't, instead of silently writing somewhere that evaporates when the runtime recycles.
set -euo pipefail

cd "$(dirname "$0")/.."

DRIVE_BACKUP_DIR="/content/drive/Shareddrives/Eric/LLM_STEER_BACKUP"
if [ ! -d "/content/drive/Shareddrives" ]; then
    echo "ERROR: /content/drive/Shareddrives not found. Mount Drive from a notebook cell first:"
    echo "  from google.colab import drive; drive.mount('/content/drive')"
    echo "(Shared Drives mount automatically under Shareddrives/ as long as your account has access"
    echo " to the shared drive -- no separate flag needed.)"
    exit 1
fi
mkdir -p "$DRIVE_BACKUP_DIR/results" "$DRIVE_BACKUP_DIR/hf_cache"

pip install --upgrade pip
pip install -r requirements.txt

# Point the HF cache at the Shared Drive, not local Colab disk. The 7B model weights (~14GB) then
# only get downloaded ONCE across all your future Colab sessions -- local disk is wiped every time
# the runtime recycles, the Shared Drive isn't.
export HF_HOME="$DRIVE_BACKUP_DIR/hf_cache"
echo "export HF_HOME=\"$DRIVE_BACKUP_DIR/hf_cache\"" >> ~/.bashrc

python3 -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
AutoTokenizer.from_pretrained(name)
<<<<<<< HEAD
AutoModelForCausalLM.from_pretrained(name)
=======
AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map='auto')
>>>>>>> b987d343167a24781f8484775ee0835a6fd05157
print('model cached to', __import__('os').environ.get('HF_HOME'))
"

echo "== setup done. Drive-backed results dir: $DRIVE_BACKUP_DIR/results =="