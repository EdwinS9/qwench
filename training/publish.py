"""Publish the trained SDFT LoRA adapter to a private HuggingFace repo.

Reads the adapter saved to the `qwench-checkpoints` Modal volume and uploads it plus a
model card to a private repo under your HF account.

One-time setup — create a dedicated Modal secret from your HF **write** token:

    modal secret create hf-secret HF_TOKEN=hf_...

Then publish (after a training run has saved an adapter):

    modal run training/publish.py                       # uploads sdft-best/ -> private repo
    modal run training/publish.py --src sdft-final
    modal run training/publish.py --repo my-custom-name --private False
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.25")
app = modal.App("qwench-publish", image=image)
ckpts = modal.Volume.from_name("qwench-checkpoints")

# Honest model card. The repo is a research reproduction; the card states exactly what was
# evaluated and what was not, so it reads as credible rather than overclaimed.
MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - sdft
  - self-distillation
  - robotics
  - llm-planner
  - qwen3
---

# Qwen3-8B · SDFT Robot Skill Planner (LoRA adapter)

A LoRA adapter for [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), trained with
**Self-Distillation Fine-Tuning (SDFT)** to turn a natural-language instruction plus a JSON
scene description into a sequence of robot **skill calls** (`navigate_to, pick, place, open,
close, push, detect, done`).

This is a faithful research reproduction of the method in *Self-Distillation Enables Continual
Learning* (Shenfeld, Damani, Hübotter & Agrawal, [arXiv:2601.19897](https://arxiv.org/abs/2601.19897)).
Code: https://github.com/EdwinS9/qwench

## Method (brief)

SDFT distills a **few-shot teacher** (the model shown one in-context demonstration) into the
**zero-shot student** (no demonstration), using an on-policy reverse-KL objective computed on
the student's own sampled plans, with an EMA teacher. The aim is to absorb in-context skill
into the weights while staying close to the base model's distribution.

## Intended use

Input: a system prompt (see the repo's `qwench/prompts.py`), the instruction, and a JSON scene
state. Output: a single JSON object `{"thinking": ..., "plan": [{"skill": ..., "args": {...}}]}`
ending in a `done` step. Qwen3 "thinking" mode should be **disabled** (`enable_thinking=False`).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{REPO_ID}")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

## Training data

Procedurally generated, fully verified (every gold plan is executed and must reach its goal):
~668 train / 132 held-out examples across four task families — PickAndPlace, Open/Close
articulated, Push, Stack. No human labels, no API. See the repo for the generator.

## Evaluation

Graded by a **symbolic world-model executor** (parse → schema → execution → goal):

- **Plan-success on the full 132-example held-out split: ≈61%** (this checkpoint).
- **MMLU (general capability): ≈0.71**, statistically unchanged from base Qwen3-8B (≈0.71) —
  i.e. **no measurable forgetting** under this LoRA setup (see caveat below).

The full 132-example split is the headline number; the 64-example in-training eval subset is
easier and scores higher. This checkpoint is saved early (at the first in-training peak), so it
under-fits the harder task families such as multi-step stacking, where most failures land.

## Limitations (read before relying on this)

- **Symbolic grader, not a physics simulator.** Plans are validated against a symbolic world
  model, not ManiSkill or a real robot. Physical executability is **not** demonstrated.
- **Templated task.** Instructions/scenes are templated; the task is comparatively easy once the
  output format is learned.
- **Forgetting not demonstrated *vs.* SFT.** Both SFT and SDFT leave MMLU essentially unchanged
  here — expected, because LoRA trains ~1% of parameters and is inherently forgetting-resistant.
  SDFT's anti-forgetting advantage is a **full fine-tuning** phenomenon and is **not** shown by
  this LoRA adapter. Treat this as a method reproduction, not a SOTA claim.
- Vocabulary is limited to the 8 skills above and the four task families.

## Citation

```bibtex
@misc{shenfeld2026selfdistillation,
  title         = {Self-Distillation Enables Continual Learning},
  author        = {Shenfeld, Idan and Damani, Mehul and H\\"ubotter, Jonas and Agrawal, Pulkit},
  year          = {2026},
  eprint        = {2601.19897},
  archivePrefix = {arXiv}
}
```
"""


@app.function(volumes={"/root/checkpoints": ckpts},
              secrets=[modal.Secret.from_name("hf-secret")])
def publish(repo: str, src: str, private: bool):
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not found in the Modal secret. Add a HF write token to .env "
            "(HF_TOKEN=hf_...) and run ./scripts/push-secret.sh, then retry."
        )
    src_dir = f"/root/checkpoints/{src}"
    if not os.path.isdir(src_dir):
        raise SystemExit(f"no adapter at {src_dir} — has a training run saved one yet?")

    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{repo}"
    api.create_repo(repo_id, private=private, repo_type="model", exist_ok=True)
    # write the card into the folder so it's uploaded alongside the adapter
    from pathlib import Path as P
    P(f"{src_dir}/README.md").write_text(MODEL_CARD.replace("{REPO_ID}", repo_id))
    api.upload_folder(folder_path=src_dir, repo_id=repo_id, repo_type="model")
    print(f"published ({'private' if private else 'public'}): https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main(repo: str = "qwen3-8b-sdft-robot-planner", src: str = "sdft-best", private: bool = True):
    publish.remote(repo, src, private)
