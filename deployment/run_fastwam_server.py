# Copyright (c) 2026 Gamma Robotics (LocoManiLab). Licensed under the MIT License; see LICENSE.
"""
Drop-in PolicyServer compatible with Isaac-GR00T's ZMQ REQ/REP protocol,
backed by a fine-tuned FastWAM (Wan2.2-based world-action model).

Lets gear_sonic/scripts/run_vla_inference.py connect to FastWAM without any
client-side change -- same wire protocol, same observation/action keys as the
lingbot-va server (wan_va/gr00t_compat_server.py), which this file mirrors.

Key differences from the lingbot-va adapter (and why):
  * FastWAM.infer_action is STATELESS. Each call takes ONE conditioning image
    + the current 43-dim proprio + the (cached) prompt embedding and denoises a
    full action chunk. There is no KV cache / frame_st_id / closed-loop machinery
    here -- replanning is open-loop chunked, driven by the client's blocking loop.
  * FastWAM predicts `action.motion_token` (64-dim learned motion latent),
    `action.smpl_joints` (72-dim SMPL joints), or -- for heft-controller
    checkpoints -- `action.ref_qpos` (36-dim) + `action.hands` (2-dim),
    depending on the checkpoint action layout / --action-space. Older sonic
    checkpoints may not predict hands; in that case this server fills them
    from --hands-mode.
  * The FastWAM custom dataset is 50 fps but trained with global_sample_stride=5,
    so successive predicted tokens are 100 ms (10 Hz) apart. The C++ controller
    consumes one action per 50 Hz tick, so this server UPSAMPLES the predicted
    chunk back to 50 Hz (see --upsample-factor / --upsample-mode) before
    returning, keeping the client/controller contract identical to lingbot-va.

Usage (single GPU, on the host that has the checkpoint + base Wan2.2 weights):
    python deployment/run_fastwam_server.py \
        --run-dir   runs/custom_1cam_1e-4/<run> \
        --fastwam-repo . \
        --initial-prompt "Push the bottle into the drawer and squat and close drawer" \
        --host 0.0.0.0 --port 5550

Wire protocol (matches gr00t.policy.server_client.PolicyServer):
    Transport : ZMQ REP socket on tcp://<host>:<port>
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config

Observation expected (subset used here -- the rest is ignored):
    obs["video"]["ego_view"]   [B=1, T, H, W, 3] uint8   (newest frame is used)
    obs["q"]                   [.., 43|29] float whole-body joint config (= proprio),
                               or obs["state"][<group>] split per anatomical group
                               (sonic: 7 groups incl. hands = 43; heft: 5 groups = 29,
                               heft's extra projected_gravity key is ignored)
    obs["state"]["left_hand"]  [.., 7]   float  (sonic only; used by --hands-mode hold)
    obs["state"]["right_hand"] [.., 7]   float
    obs["language"]["annotation.human.task_description"]   [[str]]   (optional)

Action returned (matches what run_vla_inference.py reads):
    "action.motion_token"        [H_out, 64]  float32  (motion_token mode)
    "action.smpl_joints"         [H_out, 72]  float32  (smpl mode)
    "action.left_hand_joints"    [H_out, 7]   float32  (motion_token/smpl modes)
    "action.right_hand_joints"   [H_out, 7]   float32
    "action.ref_qpos"            [H_out, 36]  float32  (heft mode; root pos+quat+29 joints)
    "action.hands"               [H_out, 2]   float32  (heft mode; open/close scalars)
    Where H_out = action_horizon * upsample_factor   (e.g. 32 * 5 = 160 @ 50Hz).
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import msgpack_numpy as mnp
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except (AttributeError, ValueError):
    pass


# ── 43-dim proprio layout (from dataset meta/modality.json) ───────────────────
# left_leg 0:6 | right_leg 6:12 | waist 12:15 | left_arm 15:22 |
# left_hand 22:29 | right_arm 29:36 | right_hand 36:43
PROPRIO_DIM = 43
LEFT_HAND_SLICE = slice(22, 29)
RIGHT_HAND_SLICE = slice(36, 43)
HAND_DIM = 7
MOTION_TOKEN_DIM = 64
SMPL_JOINTS_DIM = 72

# ── heft controller (external/heft) checkpoints ───────────────────────────────
# observation.state is 29-dim body joints only (hands are NOT folded in), split
# into the same anatomical groups minus the hand groups; the policy predicts
# ref_qpos = root_pos(3) + root_quat wxyz(4) + 29 joint targets, plus 2 hand
# open/close scalars.
HEFT_PROPRIO_GROUPS = ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]
REF_QPOS_DIM = 36
REF_QPOS_QUAT_SLICE = slice(3, 7)
HEFT_HANDS_DIM = 2


def _mixed_precision_to_dtype(mp: str) -> torch.dtype:
    key = str(mp).strip().lower()
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[key]


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def _latest_weights(run_dir: Path) -> Path:
    weights_dir = run_dir / "checkpoints" / "weights"
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"No weights dir at {weights_dir}")
    pts = sorted(weights_dir.glob("step_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No step_*.pt checkpoints in {weights_dir}")
    # step_040000.pt sorts last lexicographically (zero-padded) == newest.
    return pts[-1]


def _encode_prompt_standalone(args, cfg) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load ONLY the T5 text encoder, encode --initial-prompt, free it, and return
    (context, context_mask, prompt) on CPU.

    Mirrors FastWAM.encode_prompt but WITHOUT building the DiT/VAE, so peak VRAM is
    just T5 alone (~11 GB) instead of T5 + transformer together (~22 GB). That is
    what lets the --evict-t5 path run on a 24 GB GPU as a single launch, with no
    separate offline --dump-prompt-embeds step (same idea as the lingbot-va
    `_preencode_prompt` / --evict-t5 flow). Must be called BEFORE the model is
    built so the two never co-reside.
    """
    import gc

    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    dtype = _mixed_precision_to_dtype(args.mixed_precision)
    mcfg = cfg.model
    model_id = mcfg.get("model_id")
    tokenizer_model_id = mcfg.get("tokenizer_model_id", model_id)
    tokenizer_max_len = int(mcfg.get("tokenizer_max_len", 512))
    redirect = bool(mcfg.get("redirect_common_files", True))
    if args.base_model_path:
        base = str(Path(args.base_model_path).expanduser().resolve())
        if not Path(base).exists():
            raise FileNotFoundError(f"--base-model-path not found: {base}")
        model_id = tokenizer_model_id = base
        redirect = False  # use the local .pth T5 shipped in the Wan2.2 dir

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id, tokenizer_model_id=tokenizer_model_id, redirect_common_files=redirect,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    t0 = time.monotonic()
    print(f"[fastwam] --evict-t5: loading STANDALONE T5 from {text_config.path}", flush=True)
    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=dtype, device=args.device,
    )
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path, seq_len=tokenizer_max_len, clean="whitespace",
    )

    prompt = args.initial_prompt
    ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(args.device)
    mask = mask.to(args.device, dtype=torch.bool)
    with torch.no_grad():
        prompt_emb = text_encoder(ids, mask)             # [1, L, D]
    # Match FastWAM.encode_prompt: zero padded positions, then return all-ones mask.
    seq_lens = mask.gt(0).sum(dim=1).long()
    for i, v in enumerate(seq_lens):
        prompt_emb[i, v:] = 0
    out_mask = torch.ones_like(mask)

    context = prompt_emb.detach().to("cpu")
    context_mask = out_mask.detach().to("cpu")

    del text_encoder, tokenizer, prompt_emb, ids, mask, out_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[fastwam] --evict-t5: T5 encoded + freed in {(time.monotonic() - t0) * 1000:.0f} ms; "
        f"ctx={tuple(context.shape)} mask={tuple(context_mask.shape)} prompt={prompt!r}",
        flush=True,
    )
    return context, context_mask, prompt


def _load_cache_t5_embeds(cache_dir: str, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a TRAINING-format text-embed cache entry for `prompt` and convert it
    to serving format. Returns (context [1, L, D], context_mask [1, L]) on CPU.

    Training caches (FastWAM scripts/precompute_text_embeds.py, consumed by
    robot_video_dataset._get_cached_text_context) store one file per prompt:
        <sha256(prompt)>.t5_len<L>.<enc_id>.pt -> {"context": [L, D], "mask": [L]}
    with RAW embeddings (padding not zeroed) and the REAL attention mask, while
    serving (FastWAM.encode_prompt / _encode_prompt_standalone) wants a batched
    context with padded positions zeroed and an all-ones mask.
    """
    import hashlib

    cache = Path(cache_dir).expanduser().resolve()
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    matches = sorted(cache.glob(f"{hashed}.t5_len*.pt"))
    if not matches:
        available = "\n".join(f"  {p.name}" for p in sorted(cache.glob("*.pt"))) or "  (empty)"
        raise FileNotFoundError(
            f"No cached T5 embeds for prompt {prompt!r} in {cache} "
            f"(looked for {hashed}.t5_len*.pt).\n"
            f"The prompt must EXACTLY match a training task prompt, i.e. the full "
            f"'A video recorded from ... instruction: <task>' template string.\n"
            f"Cache contains:\n{available}"
        )
    path = matches[0]
    if len(matches) > 1:
        print(f"[fastwam] --t5-cache: {len(matches)} entries match, using {path.name}", flush=True)
    payload = torch.load(str(path), map_location="cpu")
    context = payload["context"]
    mask = payload["mask"].bool()
    if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Bad cache entry {path}: context {tuple(context.shape)}, mask {tuple(mask.shape)} "
            "(expected [L, D] + [L])"
        )
    valid = int(mask.sum())
    context = context.clone().unsqueeze(0)                       # [1, L, D]
    context[:, valid:] = 0                                       # zero padded positions
    context_mask = torch.ones(1, mask.shape[0], dtype=torch.bool)
    print(
        f"[fastwam] --t5-cache: loaded {path.name} "
        f"(ctx={tuple(context.shape)} valid_len={valid} prompt={prompt!r})",
        flush=True,
    )
    return context, context_mask


# ── FastWAM policy adapter ────────────────────────────────────────────────────
class FastWAMPolicy:
    """Wraps a FastWAM model so it answers Isaac-GR00T get_action() calls."""

    def __init__(
        self,
        model,
        processor,
        *,
        action_horizon: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        video_size_wh: tuple[int, int],
        upsample_factor: int,
        upsample_mode: str,
        hands_mode: str,
        action_space: str,
        evict_t5: bool,
    ) -> None:
        self.model = model
        self.processor = processor
        self._has_text_encoder = getattr(model, "text_encoder", None) is not None

        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.video_size_wh = video_size_wh
        self.upsample_factor = max(1, int(upsample_factor))
        self.upsample_mode = str(upsample_mode)
        self.hands_mode = str(hands_mode)
        self.requested_action_space = str(action_space)
        self.evict_t5 = bool(evict_t5)

        self._state_key = processor.shape_meta["state"][0]["key"]
        self._action_layout = [
            (m["key"], int(m["shape"])) for m in processor.shape_meta["action"]
        ]
        self._action_key = self._action_layout[0][0]
        self._action_space = self._resolve_action_space(self.requested_action_space)
        self._model_predicts_hands = {"left_hand_joints", "right_hand_joints"}.issubset(
            {k for k, _ in self._action_layout}
        )
        print(
            f"[fastwam] action layout={self._action_layout}; "
            f"resolved action_space={self._action_space}",
            flush=True,
        )

        self._current_prompt: Optional[str] = None
        self._cached_context: Optional[torch.Tensor] = None
        self._cached_context_mask: Optional[torch.Tensor] = None

    # ---- public endpoints --------------------------------------------------

    def ping(self) -> dict:
        return {"status": "ok", "message": "fastwam server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        self._set_prompt(prompt)
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        t0 = time.monotonic()

        prompt = self._extract_prompt(observation)
        if prompt is not None and prompt != self._current_prompt:
            if not self._has_text_encoder:
                # Precomputed-embeds mode: text encoder isn't loaded, so we can't
                # re-encode. Keep serving the cached prompt and warn once.
                if not getattr(self, "_warned_locked_prompt", False):
                    print(
                        f"[fastwam] ignoring prompt change to {prompt!r}: running with "
                        f"precomputed embeds for {self._current_prompt!r} (no text encoder loaded)",
                        flush=True,
                    )
                    self._warned_locked_prompt = True
            else:
                print(f"[fastwam] prompt changed -> re-encode. new={prompt!r}", flush=True)
                self._set_prompt(prompt)
        if self._current_prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt at startup, call reset(prompt=...), "
                "or include obs['language']['annotation.human.task_description'][0][0]."
            )

        image_tensor = self._build_image_tensor(observation)      # [1,3,H,W] in [-1,1]
        raw_proprio = self._extract_proprio(observation)          # [43|29] raw (un-normalized)
        proprio = self._normalize_state(raw_proprio)              # [1,43|29] normalized

        infer_kwargs = dict(
            prompt=None,
            input_image=image_tensor,
            action_horizon=self.action_horizon,
            proprio=proprio,
            context=self._cached_context,
            context_mask=self._cached_context_mask,
            negative_prompt=self.negative_prompt,
            text_cfg_scale=self.text_cfg_scale,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            seed=self.seed,
            rand_device=self.rand_device,
            tiled=self.tiled,
        )

        t_inf = time.monotonic()
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        action_norm = pred["action"]                              # [T, sum_dim] normalized
        denorm = self._denormalize_action(action_norm)            # {key: [T, dim]} raw
        primary_action = self._upsample(denorm[self._action_key])
        h_out = primary_action.shape[0]

        if self._action_space == "heft":
            # heft models predict the 2-dim open/close "hands" key themselves;
            # the sonic 7-dof hand-joint machinery below does not apply.
            left_hand = right_hand = None
            hands_src = "predicted"
        elif self._model_predicts_hands:
            # Fine-tuned model predicts hand joints directly -> use them
            # (--hands-mode then only applies to legacy 64-dim models).
            left_hand = self._upsample(denorm["left_hand_joints"])
            right_hand = self._upsample(denorm["right_hand_joints"])
            hands_src = "predicted"
        else:
            left_hand, right_hand = self._build_hands(raw_proprio, h_out)
            hands_src = self.hands_mode

        print(
            f"[fastwam] get_action: T={self.action_horizon} -> H_out={h_out} "
            f"(x{self.upsample_factor} {self.upsample_mode}); infer={infer_ms:.0f} ms; "
            f"action_space={self._action_space}; hands={hands_src}; prompt={self._current_prompt!r}",
            flush=True,
        )

        if self._action_space == "motion_token":
            if primary_action.shape[-1] != MOTION_TOKEN_DIM:
                raise RuntimeError(
                    f"motion_token action must be {MOTION_TOKEN_DIM}D, got {primary_action.shape}"
                )
            action_dict = {
                "action.motion_token": np.ascontiguousarray(primary_action, dtype=np.float32),
                "action.left_hand_joints": np.ascontiguousarray(left_hand, dtype=np.float32),
                "action.right_hand_joints": np.ascontiguousarray(right_hand, dtype=np.float32),
            }
        elif self._action_space == "smpl":
            if primary_action.shape[-1] != SMPL_JOINTS_DIM:
                raise RuntimeError(
                    f"smpl_joints action must be {SMPL_JOINTS_DIM}D, got {primary_action.shape}"
                )
            action_dict = {
                "action.smpl_joints": np.ascontiguousarray(primary_action, dtype=np.float32),
                "action.left_hand_joints": np.ascontiguousarray(left_hand, dtype=np.float32),
                "action.right_hand_joints": np.ascontiguousarray(right_hand, dtype=np.float32),
            }
        elif self._action_space == "heft":
            if primary_action.shape[-1] != REF_QPOS_DIM:
                raise RuntimeError(
                    f"ref_qpos action must be {REF_QPOS_DIM}D, got {primary_action.shape}"
                )
            # linear upsampling of the root quat (cols 3:7) does not preserve unit
            # norm -> renormalize so the heft controller gets valid orientations.
            quat = primary_action[:, REF_QPOS_QUAT_SLICE]
            norms = np.linalg.norm(quat, axis=-1, keepdims=True)
            primary_action[:, REF_QPOS_QUAT_SLICE] = quat / np.clip(norms, 1e-8, None)
            if "hands" in {k for k, _ in self._action_layout}:
                hands = self._upsample(denorm["hands"])
            else:
                hands = np.zeros((h_out, HEFT_HANDS_DIM), dtype=np.float32)
            # The sim's RefQposAdapter (g1_backend/sim2real/vla/adapters.py) requires the
            # reference frame as THREE separate keys, not one combined ref_qpos array, and
            # 7+7 hand joint targets rather than a 2-D open/close pair. Emit both shapes:
            # the split keys for the sim2real stack, and the combined ones the real-robot
            # C++ consumer (gear_sonic run_vla_inference.py) already reads. Extra keys are
            # ignored by whichever side doesn't want them.
            action_dict = {
                "action.ref_root_pos": np.ascontiguousarray(primary_action[:, 0:3], dtype=np.float32),
                "action.ref_root_quat": np.ascontiguousarray(primary_action[:, 3:7], dtype=np.float32),
                "action.ref_joint_pos": np.ascontiguousarray(primary_action[:, 7:REF_QPOS_DIM], dtype=np.float32),
                "action.ref_qpos": np.ascontiguousarray(primary_action, dtype=np.float32),
                "action.hands": np.ascontiguousarray(hands, dtype=np.float32),
            }
            if hands.shape[-1] == 14:      # Dex3 joints: hand targets the sim can apply
                action_dict["action.left_hand_joints"] = np.ascontiguousarray(
                    hands[:, :7], dtype=np.float32)
                action_dict["action.right_hand_joints"] = np.ascontiguousarray(
                    hands[:, 7:], dtype=np.float32)
        else:
            raise RuntimeError(f"Unsupported FastWAM action_space: {self._action_space}")
        info = {
            "inference_time_s": time.monotonic() - t0,
            "chunk_horizon": int(h_out),
            "action_space": self._action_space,
        }
        return action_dict, info

    # ---- internals ---------------------------------------------------------

    def _resolve_action_space(self, requested: str) -> str:
        requested = (requested or "auto").strip().lower()
        if self._action_key == "motion_token":
            inferred = "motion_token"
            expected_dim = MOTION_TOKEN_DIM
        elif self._action_key == "smpl_joints":
            inferred = "smpl"
            expected_dim = SMPL_JOINTS_DIM
        elif self._action_key == "ref_qpos":
            inferred = "heft"
            expected_dim = REF_QPOS_DIM
        else:
            raise ValueError(
                f"Unsupported FastWAM primary action key {self._action_key!r}; "
                "expected 'motion_token', 'smpl_joints' or 'ref_qpos'."
            )

        actual_dim = dict(self._action_layout)[self._action_key]
        if actual_dim != expected_dim:
            raise ValueError(
                f"FastWAM primary action {self._action_key!r} has dim {actual_dim}, "
                f"expected {expected_dim} for action_space={inferred}."
            )

        if requested == "auto":
            return inferred
        if requested not in {"motion_token", "smpl", "heft"}:
            raise ValueError("--action-space must be one of: auto, motion_token, smpl, heft")
        if requested != inferred:
            raise ValueError(
                f"--action-space={requested} does not match checkpoint primary action "
                f"{self._action_key!r} (inferred {inferred})."
            )
        return requested

    def set_prompt_embeds(self, context: torch.Tensor, context_mask: torch.Tensor, prompt: str) -> None:
        """Inject precomputed prompt embeds (used when the text encoder isn't loaded)."""
        self._cached_context = context.detach()
        self._cached_context_mask = context_mask.detach()
        self._current_prompt = prompt
        print(
            f"[fastwam] using precomputed embeds: ctx={tuple(context.shape)} "
            f"mask={tuple(context_mask.shape)} prompt={prompt!r}",
            flush=True,
        )

    def _set_prompt(self, prompt: Optional[str]) -> None:
        if prompt is None:
            prompt = self._current_prompt
        if prompt is None:
            return
        if not self._has_text_encoder:
            # Precomputed / standalone-evicted embeds: the model has no text encoder,
            # so we cannot re-encode. Keep serving the cached prompt (don't crash).
            if self._current_prompt is not None and prompt != self._current_prompt:
                print(
                    f"[fastwam] reset: ignoring prompt {prompt!r}; locked on cached embeds "
                    f"for {self._current_prompt!r} (no text encoder loaded)",
                    flush=True,
                )
            return
        if self.evict_t5 and self._current_prompt is not None and prompt != self._current_prompt:
            raise RuntimeError(
                f"--evict-t5: text encoder was freed after encoding {self._current_prompt!r}; "
                f"cannot encode new prompt {prompt!r}. Restart the server with --initial-prompt."
            )
        context, context_mask = self.model.encode_prompt(prompt)
        self._cached_context = context.detach()
        self._cached_context_mask = context_mask.detach()
        self._current_prompt = prompt
        print(f"[fastwam] prompt encoded: ctx={tuple(context.shape)} mask={tuple(context_mask.shape)}", flush=True)
        if self.evict_t5:
            self.model.text_encoder = None
            self.model.tokenizer = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[fastwam] --evict-t5: text encoder/tokenizer freed", flush=True)

    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        lang = obs.get("language") or {}
        raw = lang.get("annotation.human.task_description")
        if raw is None:
            return None
        while isinstance(raw, (list, tuple)) and len(raw) > 0:
            if isinstance(raw[0], (list, tuple)):
                raw = raw[0]
            else:
                break
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw if isinstance(raw, str) else None

    def _extract_image_stack(self, obs: dict) -> np.ndarray:
        """Return ego_view image stack as [T, H, W, 3] uint8 (newest last)."""
        video = obs.get("video") or {}
        img = None
        for k in ("ego_view", "observation.images.ego_view"):
            if k in video:
                img = video[k]
                break
        if img is None:
            if len(video) == 1:
                img = next(iter(video.values()))
            else:
                raise KeyError(f"No ego_view in obs['video']; keys={list(video.keys())}")
        img = np.asarray(img)
        while img.ndim > 4 and img.shape[0] == 1:
            img = img[0]
        if img.ndim == 3:
            img = img[None]
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"ego_view must end in [..,H,W,3], got {img.shape}")
        return img.astype(np.uint8)

    def _build_image_tensor(self, obs: dict) -> torch.Tensor:
        stack = self._extract_image_stack(obs)        # [T,H,W,3]
        frame = stack[-1]                              # newest frame [H,W,3]
        resized = _resize_rgb(frame, self.video_size_wh)  # [Hs,Ws,3]
        t = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        return t * (2.0 / 255.0) - 1.0                 # [1,3,Hs,Ws] in [-1,1]

    def _extract_proprio(self, obs: dict) -> np.ndarray:
        """Return the raw whole-body joint config (= dataset observation.state).

        sonic checkpoints: 43-dim (hand joints folded in). heft checkpoints: 29-dim
        (body joints only; heft state groups carry no hands, and the extra
        projected_gravity key the heft client sends is not part of the state).
        """
        expected = int(self.processor.shape_meta["state"][0]["raw_shape"])
        q = obs.get("q")
        if q is not None:
            q = np.asarray(q, dtype=np.float32).reshape(-1)
            if q.shape[0] != expected:
                raise ValueError(f"obs['q'] must be {expected}-dim, got {q.shape[0]}")
            return q
        # Fallback: reconstruct from split state groups in modality order.
        state = obs.get("state") or {}
        if self._action_space == "heft":
            order = HEFT_PROPRIO_GROUPS
        else:
            order = ["left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand"]
        parts = []
        for g in order:
            if g not in state:
                raise KeyError(f"obs has neither 'q' nor obs['state']['{g}']")
            parts.append(np.asarray(state[g], dtype=np.float32).reshape(-1))
        q = np.concatenate(parts)
        if q.shape[0] != expected:
            raise ValueError(f"Reconstructed proprio is {q.shape[0]}-dim, expected {expected}")
        return q

    def _normalize_state(self, raw_state: np.ndarray) -> torch.Tensor:
        batch = {"state": {self._state_key: torch.as_tensor(raw_state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self._state_key]        # [1,43]

    def _denormalize_action(self, action: torch.Tensor) -> dict:
        # Split the merged [T, sum_dim] action and denormalize each key separately.
        # The model emits normalized per-key values concatenated in
        # shape_meta["action"] order (ConcatLeftAlign): for the 78-dim model that
        # is motion_token[64] + left_hand_joints[7] + right_hand_joints[7].
        if action.ndim == 2:
            action = action.unsqueeze(0)               # [1,T,D]
        action = action.to(dtype=torch.float32, device="cpu")
        out = {}
        idx = 0
        for key, dim in self._action_layout:
            seg = action[:, :, idx:idx + dim]
            normalizer = self.processor.normalizer.normalizers["action"][key]
            out[key] = normalizer.backward(seg)[0].numpy()   # [T, dim]
            idx += dim
        return out

    def _upsample(self, tokens: np.ndarray) -> np.ndarray:
        """Upsample [T,D] predicted at 10 Hz to [T*f,D] at 50 Hz."""
        f = self.upsample_factor
        if f <= 1:
            return tokens
        if self.upsample_mode == "repeat":
            return np.repeat(tokens, f, axis=0)
        # linear interpolation between consecutive keyframe tokens
        T, D = tokens.shape
        out_len = T * f
        src = np.arange(T, dtype=np.float64)
        tgt = np.clip(np.arange(out_len, dtype=np.float64) / f, 0, T - 1)
        out = np.empty((out_len, D), dtype=np.float32)
        for d in range(D):
            out[:, d] = np.interp(tgt, src, tokens[:, d])
        return out

    def _build_hands(self, raw_proprio: np.ndarray, h_out: int) -> tuple[np.ndarray, np.ndarray]:
        if self.hands_mode == "zeros":
            z = np.zeros((h_out, HAND_DIM), dtype=np.float32)
            return z, z.copy()
        # hold: repeat the currently observed hand joints across the chunk
        left = raw_proprio[LEFT_HAND_SLICE].astype(np.float32)
        right = raw_proprio[RIGHT_HAND_SLICE].astype(np.float32)
        return np.tile(left, (h_out, 1)), np.tile(right, (h_out, 1))


# ── ZMQ server (mirrors gr00t.policy.server_client.PolicyServer wire format) ─
def serve(policy: FastWAMPolicy, host: str, port: int) -> None:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[fastwam] PolicyServer ready on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                running = False
                resp = {"status": "killing"}
            elif endpoint == "get_action":
                action, info = policy.get_action(
                    observation=data.get("observation", {}),
                    options=data.get("options"),
                )
                resp = [action, info]
            elif endpoint == "reset":
                resp = policy.reset(options=data.get("options"))
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            else:
                resp = {"error": f"Unknown endpoint: {endpoint}"}

            sock.send(mnp.packb(resp))
        except zmq.error.ContextTerminated:
            break
        except Exception as e:
            print(f"[fastwam] server error: {e}\n{traceback.format_exc()}", flush=True)
            try:
                sock.send(mnp.packb({"error": str(e)}))
            except Exception:
                pass

    try:
        sock.close(linger=0)
    finally:
        ctx.term()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_processor(args):
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = OmegaConf.load(run_dir / "config.yaml")

    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _latest_weights(run_dir)
    stats_path = Path(args.dataset_stats).expanduser().resolve() if args.dataset_stats else (run_dir / "dataset_stats.json")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not stats_path.exists():
        raise FileNotFoundError(f"dataset_stats.json not found: {stats_path}")

    model_dtype = _mixed_precision_to_dtype(args.mixed_precision)

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    # Load the T5 text encoder INSIDE the model only when the model itself must
    # encode prompts at runtime. We build DiT/VAE without T5 when:
    #   * --prompt-embeds : embedding precomputed offline, or
    #   * --t5-cache     : embedding read from the dataset's training cache, or
    #   * --evict-t5      : main() pre-encodes with a STANDALONE T5 (loaded + freed
    #                       before this model is built), so T5 never co-resides with
    #                       the transformer -- this is what fits a 24 GB GPU.
    # --dump-prompt-embeds is the exception: it needs T5 in the model to encode.
    model_cfg.load_text_encoder = bool(args.dump_prompt_embeds) or (
        (args.prompt_embeds is None) and (args.t5_cache is None) and (not args.evict_t5)
    )
    # Inference overrides (match sim_libero.yaml / sim_robotwin.yaml): the full
    # MoT weights come from load_checkpoint, so skip the redundant pretrain init
    # and don't require the action_dit backbone file.
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    # Override the base Wan2.2 weights dir (VAE + text_encoder + tokenizer) so the
    # run config's absolute remote path doesn't have to exist locally.
    if args.base_model_path:
        base = str(Path(args.base_model_path).expanduser().resolve())
        if not Path(base).exists():
            raise FileNotFoundError(f"--base-model-path not found: {base}")
        model_cfg.model_id = base
        model_cfg.tokenizer_model_id = base
        print(f"[fastwam] base model path overridden -> {base}", flush=True)
        # All-local: use the original .pth T5/VAE shipped in the Wan2.2 dir instead
        # of redirecting to the DiffSynth converted safetensors (which would trigger a
        # download). The .pth hashes are the ones in WAN22_MODEL_REGISTRY.
        if "redirect_common_files" in model_cfg:
            model_cfg.redirect_common_files = False
            print("[fastwam] redirect_common_files=False (use local .pth T5/VAE)", flush=True)
    print(f"[fastwam] instantiating model (id={model_cfg.get('model_id')})", flush=True)
    model = instantiate(model_cfg, model_dtype=model_dtype, device=args.device)
    print(f"[fastwam] loading checkpoint: {checkpoint}", flush=True)
    model.load_checkpoint(str(checkpoint))
    model = model.to(args.device).eval()

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

    video_size = list(cfg.data.train.video_size)          # [H, W]
    video_size_wh = (int(video_size[1]), int(video_size[0]))  # PIL wants (W, H)
    num_frames = int(cfg.data.train.num_frames)
    default_horizon = num_frames - 1

    return model, processor, video_size_wh, default_horizon


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FastWAM policy server (Gr00t-compatible).")
    p.add_argument("--run-dir", required=True,
                   help="Training run dir containing config.yaml + dataset_stats.json + checkpoints/")
    p.add_argument("--fastwam-repo", default=None,
                   help="Path to the FastWAM repo (its src/ is prepended to sys.path)")
    p.add_argument("--base-model-path", default=None,
                   help="Base Wan2.2-TI2V-5B dir (VAE + text_encoder + tokenizer). Overrides "
                        "model_id/tokenizer_model_id from the run config (use when the config's "
                        "absolute path doesn't exist on this machine).")
    p.add_argument("--checkpoint", default=None,
                   help="weights/step_XXXXX.pt (default: newest in <run-dir>/checkpoints/weights)")
    p.add_argument("--dataset-stats", default=None,
                   help="dataset_stats.json (default: <run-dir>/dataset_stats.json)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--initial-prompt", default=None,
                   help="Pre-encode this prompt so get_action works without language in obs")
    # inference
    p.add_argument("--action-horizon", type=int, default=None,
                   help="Predicted tokens per chunk (default: num_frames-1 from config)")
    p.add_argument("--action-space", default="auto", choices=["auto", "motion_token", "smpl", "heft"],
                   help="Policy output type. auto infers from checkpoint action key; "
                        "motion_token returns action.motion_token; smpl returns action.smpl_joints; "
                        "heft returns action.ref_qpos + action.hands (heft controller checkpoints).")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--rand-device", default="cpu")
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--evict-t5", action="store_true",
                   help="Load a STANDALONE T5, encode --initial-prompt, free it, THEN build the "
                        "model (T5 never co-resident). Fits a 24 GB GPU in a single launch -- no "
                        "offline --dump-prompt-embeds needed. Prompt is fixed for the session.")
    p.add_argument("--prompt-embeds", default=None,
                   help="Serve from a precomputed prompt-embeds .pt (skips loading T5 entirely; "
                        "recommended on 24 GB GPUs). Build it once with --dump-prompt-embeds.")
    p.add_argument("--t5-cache", default=None,
                   help="Serve from a TRAINING-format text-embed cache dir (the dataset's "
                        "cache_t5/, fetched next to the run by fetch_fastwam_ckpt.sh). Looks up "
                        "sha256(--initial-prompt) and skips loading T5 entirely. Requires "
                        "--initial-prompt; --prompt-embeds wins if both are given.")
    p.add_argument("--dump-prompt-embeds", default=None,
                   help="Encode --initial-prompt and save {context, context_mask, prompt} to this "
                        ".pt file, then exit. Run on the remote GPU or locally with --device cpu "
                        "(loading T5 + transformer needs ~22 GB of RAM/VRAM together).")
    # 10Hz -> 50Hz upsampling for the 50Hz C++ controller
    p.add_argument("--upsample-factor", type=int, default=5,
                   help="Predicted tokens are 100ms apart; x5 -> 50Hz. Set 1 to disable.")
    p.add_argument("--upsample-mode", default="interp", choices=["interp", "repeat"])
    # hands (FastWAM does not predict them)
    p.add_argument("--hands-mode", default="hold", choices=["hold", "zeros"],
                   help="hold: repeat observed hand joints; zeros: emit np.zeros(7)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    repo = args.fastwam_repo or os.environ.get("FASTWAM_REPO")
    if repo:
        src = str(Path(repo).expanduser().resolve() / "src")
        for path in (str(Path(repo).expanduser().resolve()), src):
            if path not in sys.path:
                sys.path.insert(0, path)

    # ── --evict-t5: pre-encode the prompt with a STANDALONE T5 (loaded + freed
    #    HERE, before the DiT/VAE are built) so T5 never co-resides with the
    #    transformer. This is what makes --evict-t5 fit on a 24 GB GPU without a
    #    separate offline --dump-prompt-embeds step. Dump mode keeps T5 inside the
    #    model, so it is excluded here.
    pre_embeds = None
    if args.evict_t5 and not args.prompt_embeds and not args.t5_cache and not args.dump_prompt_embeds:
        if not args.initial_prompt:
            raise SystemExit("--evict-t5 requires --initial-prompt (T5 is pre-encoded then freed)")
        cfg_for_t5 = OmegaConf.load(Path(args.run_dir).expanduser().resolve() / "config.yaml")
        pre_embeds = _encode_prompt_standalone(args, cfg_for_t5)

    # ── --t5-cache: read the prompt's embedding from the dataset's training cache
    #    (no T5 at all). Done BEFORE the model is built so a missing cache entry
    #    fails in seconds, not after the 12 GB checkpoint load.
    cache_embeds = None
    if args.t5_cache and not args.prompt_embeds and not args.dump_prompt_embeds:
        if not args.initial_prompt:
            raise SystemExit("--t5-cache requires --initial-prompt (it is the cache lookup key)")
        cache_embeds = _load_cache_t5_embeds(args.t5_cache, args.initial_prompt)

    model, processor, video_size_wh, default_horizon = load_model_and_processor(args)
    action_horizon = args.action_horizon or default_horizon

    # ── Dump mode: encode the prompt, save embeds, exit (run where there's RAM/VRAM
    #    for T5 + transformer together -- remote GPU, or locally with --device cpu).
    if args.dump_prompt_embeds:
        if not args.initial_prompt:
            raise SystemExit("--dump-prompt-embeds requires --initial-prompt")
        context, context_mask = model.encode_prompt(args.initial_prompt)
        out = Path(args.dump_prompt_embeds).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"context": context.detach().cpu(),
             "context_mask": context_mask.detach().cpu(),
             "prompt": args.initial_prompt},
            str(out),
        )
        print(f"[fastwam] saved prompt embeds -> {out} (prompt={args.initial_prompt!r})", flush=True)
        return

    policy = FastWAMPolicy(
        model,
        processor,
        action_horizon=action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        rand_device=args.rand_device,
        tiled=args.tiled,
        video_size_wh=video_size_wh,
        upsample_factor=args.upsample_factor,
        upsample_mode=args.upsample_mode,
        hands_mode=args.hands_mode,
        action_space=args.action_space,
        evict_t5=args.evict_t5,
    )
    print(
        f"[fastwam] ready: horizon={action_horizon} x{args.upsample_factor} "
        f"({args.upsample_mode}) -> {action_horizon * max(1, args.upsample_factor)} actions/chunk; "
        f"steps={args.num_inference_steps}; img(WxH)={video_size_wh}; "
        f"action_space={policy._action_space}; hands={args.hands_mode}",
        flush=True,
    )

    if args.prompt_embeds:
        blob = torch.load(str(Path(args.prompt_embeds).expanduser().resolve()), map_location="cpu")
        ctx = blob["context"].to(device=model.device, dtype=model.torch_dtype)
        mask = blob["context_mask"].to(device=model.device, dtype=torch.bool)
        policy.set_prompt_embeds(ctx, mask, blob.get("prompt", "<precomputed>"))
    elif cache_embeds is not None:
        ctx, mask = cache_embeds
        policy.set_prompt_embeds(
            ctx.to(device=model.device, dtype=model.torch_dtype),
            mask.to(device=model.device, dtype=torch.bool),
            args.initial_prompt,
        )
    elif pre_embeds is not None:
        # --evict-t5: inject the embeds we computed with the standalone T5 above.
        ctx, mask, pr = pre_embeds
        policy.set_prompt_embeds(
            ctx.to(device=model.device, dtype=model.torch_dtype),
            mask.to(device=model.device, dtype=torch.bool),
            pr,
        )
    elif args.initial_prompt:
        policy.reset(options={"prompt": args.initial_prompt})

    try:
        serve(policy, args.host, args.port)
    except KeyboardInterrupt:
        print("[fastwam] interrupted, shutting down.", flush=True)


if __name__ == "__main__":
    main()
