"""User-friendly TokAN wav-to-wav inference.

TokAN (Token-based Accent Normalization / token-to-token speech LM) takes a
source speech waveform and a reference speaker waveform, and produces a new
waveform that preserves the linguistic content of the source while rendering
it in the speaker / style of the reference (or, if no reference is given,
the source speaker itself).

Pipeline:
    src wav -> WavLM + VQ -> source tokens
    source tokens -> ARLM (TransformerLM) -> target tokens
    target tokens (deduplicated) -> flow-matching decoder -> Mel
    Mel -> HiFT vocoder -> output waveform
    reference wav -> Resemblyzer -> speaker embedding (256d)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from hyperpyyaml import load_hyperpyyaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_FILENAME = "tokan.model-only.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="User-friendly TokAN wav-to-wav inference."
    )

    parser.add_argument(
        "--source_wav", required=True, help="Path to source speech wav."
    )
    parser.add_argument(
        "--reference_wav",
        default=None,
        help="Path to reference speaker wav. If omitted, source_wav is used.",
    )
    parser.add_argument(
        "--output_wav", required=True, help="Path to save generated wav."
    )

    parser.add_argument(
        "--model_config",
        default=None,
        help="Local config yaml path. If omitted, the bundled config is downloaded from HF.",
    )
    parser.add_argument(
        "--lm_checkpoint",
        default=None,
        help="Local path to ARLM checkpoint (.pt). If omitted, downloaded from HF.",
    )
    parser.add_argument(
        "--quantizer_checkpoint",
        default=None,
        help="Local path to joint quantizer + flow decoder checkpoint (.pt). "
        "If omitted, downloaded from HF.",
    )
    parser.add_argument(
        "--hift_checkpoint",
        default=None,
        help="Local path to HiFT checkpoint (.pt). If omitted, downloaded from HF.",
    )

    parser.add_argument(
        "--hf_repo_id",
        default="Piping/TokAN",
        help="Hugging Face model repo id used when checkpoint paths are not provided.",
    )
    parser.add_argument(
        "--hf_subfolder",
        default="checkpoints",
        help="Checkpoint subfolder in the HF repo.",
    )
    parser.add_argument(
        "--model_tag",
        default="default",
        help="Model bundle folder under hf_subfolder.",
    )
    parser.add_argument("--lm_filename", default="arlm.pt", help="ARLM checkpoint filename.")
    parser.add_argument(
        "--quantizer_filename",
        default="quantizer.pt",
        help="Joint quantizer + flow decoder checkpoint filename.",
    )
    parser.add_argument("--hift_filename", default="hift.pt", help="HiFT checkpoint filename.")
    parser.add_argument(
        "--config_filename",
        default=DEFAULT_CONFIG_FILENAME,
        help="Config yaml filename inside model_tag folder.",
    )
    parser.add_argument(
        "--hf_revision", default=None, help="Optional HF revision (branch/tag/commit)."
    )
    parser.add_argument("--hf_token", default=None, help="Optional Hugging Face token.")
    parser.add_argument(
        "--cache_dir",
        default="checkpoints",
        help="Local cache directory for checkpoints.",
    )

    # Generation hyper-parameters
    parser.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="Beam size for ARLM token decoding (set 0 for sampling).",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=32,
        help="Number of flow-matching steps for the speech decoder.",
    )
    parser.add_argument(
        "--full_cfg",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale for the null branch of the flow decoder.",
    )
    parser.add_argument(
        "--cond_cfg",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for the condition branch.",
    )
    parser.add_argument(
        "--spk_cfg",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for the speaker branch.",
    )
    parser.add_argument(
        "--preserve_total_duration",
        action="store_true",
        help="If set, force the synthesized waveform to match the duration of the source.",
    )
    parser.add_argument(
        "--use_lm_duration",
        action="store_true",
        help="If set, reuse the LM-emitted token repeats as durations (otherwise the "
        "flow-decoder's duration predictor is used).",
    )

    # I/O
    parser.add_argument(
        "--mel_path",
        default=None,
        help="Optional path to save the predicted Mel tensor (.pt).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hf_download(
    *,
    repo_id: str,
    subfolder: str,
    model_tag: str,
    filename: str,
    cache_dir: str,
    revision: Optional[str],
    token: Optional[str],
) -> str:
    bundle_subfolder = f"{subfolder}/{model_tag}" if subfolder else model_tag
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=bundle_subfolder,
        local_dir=cache_dir,
        revision=revision,
        token=token,
    )


def _resolve_or_download(
    local_path: Optional[str],
    *,
    repo_id: str,
    subfolder: str,
    model_tag: str,
    filename: str,
    cache_dir: str,
    revision: Optional[str],
    token: Optional[str],
) -> str:
    if local_path is not None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")
        return local_path
    return _hf_download(
        repo_id=repo_id,
        subfolder=subfolder,
        model_tag=model_tag,
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
    )


def _load_audio_as_mono(path: str, target_sr: int) -> torch.Tensor:
    """Load `path` as mono float32 tensor of shape (T,) at `target_sr`."""
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.squeeze(0)


def _extract_spk_embedding(reference_wav: str, device: torch.device) -> torch.Tensor:
    """Extract a 256-d L2-normalized speaker embedding via Resemblyzer."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except Exception as exc:
        raise RuntimeError(
            "Failed to import resemblyzer. Please `pip install resemblyzer`."
        ) from exc

    encoder_device = "cuda" if device.type == "cuda" else "cpu"
    encoder = VoiceEncoder(device=encoder_device)
    preprocessed = preprocess_wav(reference_wav)
    emb = encoder.embed_utterance(preprocessed)
    emb_t = torch.from_numpy(np.asarray(emb, dtype=np.float32)).unsqueeze(0)
    emb_t = F.normalize(emb_t, p=2, dim=-1)
    return emb_t.to(device)


def _load_models(
    config_path: str,
    lm_ckpt_path: str,
    quantizer_ckpt_path: str,
    hift_ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, int]:
    """Construct the LM / quantizer / HiFT from a HyperPyYAML config and load checkpoints."""
    with open(config_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(f)

    # ARLM
    lm = configs["lm"].to(device).eval()
    lm_ckpt = torch.load(lm_ckpt_path, map_location="cpu")
    for key in ("epoch", "step"):
        lm_ckpt.pop(key, None)
    lm.load_state_dict(lm_ckpt, strict=True)
    for p in lm.parameters():
        p.requires_grad = False

    # Joint quantizer + flow-matching speech decoder
    quantizer = configs["quantizer"].to(device).eval()
    quantizer_ckpt = torch.load(quantizer_ckpt_path, map_location="cpu")
    for key in ("epoch", "step"):
        quantizer_ckpt.pop(key, None)
    quantizer.load_state_dict(quantizer_ckpt, strict=True)
    for p in quantizer.parameters():
        p.requires_grad = False

    # HiFT vocoder
    hift = configs["hift"].to(device).eval()
    hift.load_state_dict(torch.load(hift_ckpt_path, map_location="cpu"), strict=True)
    for p in hift.parameters():
        p.requires_grad = False

    sample_rate = int(configs["sample_rate"])
    return lm, quantizer, hift, sample_rate


def _deduplicate(tokens: list[int]) -> Tuple[list[int], list[int]]:
    """Run-length encode a list of token ids -> (unique tokens, durations)."""
    if not tokens:
        return [], []
    out_tokens: list[int] = []
    out_dur: list[int] = []
    prev = tokens[0]
    count = 1
    for t in tokens[1:]:
        if t == prev:
            count += 1
        else:
            out_tokens.append(prev)
            out_dur.append(count)
            prev = t
            count = 1
    out_tokens.append(prev)
    out_dur.append(count)
    return out_tokens, out_dur


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    print(f"[TokAN] Using device: {device}")

    # ---- Resolve checkpoints (download from HF if needed) ----
    lm_ckpt = _resolve_or_download(
        args.lm_checkpoint,
        repo_id=args.hf_repo_id,
        subfolder=args.hf_subfolder,
        model_tag=args.model_tag,
        filename=args.lm_filename,
        cache_dir=args.cache_dir,
        revision=args.hf_revision,
        token=args.hf_token,
    )
    quantizer_ckpt = _resolve_or_download(
        args.quantizer_checkpoint,
        repo_id=args.hf_repo_id,
        subfolder=args.hf_subfolder,
        model_tag=args.model_tag,
        filename=args.quantizer_filename,
        cache_dir=args.cache_dir,
        revision=args.hf_revision,
        token=args.hf_token,
    )
    hift_ckpt = _resolve_or_download(
        args.hift_checkpoint,
        repo_id=args.hf_repo_id,
        subfolder=args.hf_subfolder,
        model_tag=args.model_tag,
        filename=args.hift_filename,
        cache_dir=args.cache_dir,
        revision=args.hf_revision,
        token=args.hf_token,
    )

    if args.model_config is not None:
        if not os.path.isfile(args.model_config):
            raise FileNotFoundError(f"Config file not found: {args.model_config}")
        config_path = args.model_config
    else:
        config_path = _hf_download(
            repo_id=args.hf_repo_id,
            subfolder=args.hf_subfolder,
            model_tag=args.model_tag,
            filename=args.config_filename,
            cache_dir=args.cache_dir,
            revision=args.hf_revision,
            token=args.hf_token,
        )

    print(f"[TokAN] Config: {config_path}")
    print(f"[TokAN] ARLM ckpt: {lm_ckpt}")
    print(f"[TokAN] Quantizer ckpt: {quantizer_ckpt}")
    print(f"[TokAN] HiFT ckpt: {hift_ckpt}")

    lm, quantizer, hift, sample_rate = _load_models(
        config_path, lm_ckpt, quantizer_ckpt, hift_ckpt, device
    )

    # ---- Source wav -> source token ids ----
    src_wav_16k = _load_audio_as_mono(args.source_wav, target_sr=16000).to(device)
    src_wav_16k = src_wav_16k.unsqueeze(0)  # (1, T_wav)
    src_wav_len = torch.tensor([src_wav_16k.shape[1]], dtype=torch.long, device=device)

    _, src_token, src_token_len = quantizer.quantize(src_wav_16k, src_wav_len)
    print(f"[TokAN] Extracted {src_token_len.item()} source tokens.")

    # ---- Reference wav -> speaker embedding ----
    reference_wav = args.reference_wav if args.reference_wav else args.source_wav
    spk_embed = _extract_spk_embedding(reference_wav, device=device)

    # ---- ARLM: source tokens -> target tokens ----
    token_list = []
    for tok in lm.inference(
        src_tokens=src_token,
        src_token_len=src_token_len,
        beam_size=args.beam_size,
    ):
        token_list.append(int(tok))

    if not token_list:
        raise RuntimeError("ARLM produced an empty token sequence.")
    print(f"[TokAN] ARLM emitted {len(token_list)} target tokens.")

    # ---- Prepare target tokens for the flow-matching decoder ----
    if quantizer.flow.duration_predictor is not None:
        # Decoder expects deduplicated tokens; durations are either provided or predicted.
        tgt_token_list, lm_durations = _deduplicate(token_list)
        tgt_token = torch.tensor(tgt_token_list, dtype=torch.long, device=device).unsqueeze(0)
        tgt_token_len = torch.tensor([tgt_token.shape[1]], dtype=torch.long, device=device)
        if args.use_lm_duration:
            duration = torch.tensor(lm_durations, dtype=torch.long, device=device).unsqueeze(0)
        else:
            duration = None
    else:
        tgt_token = torch.tensor(token_list, dtype=torch.long, device=device).unsqueeze(0)
        tgt_token_len = torch.tensor([tgt_token.shape[1]], dtype=torch.long, device=device)
        duration = torch.ones_like(tgt_token)

    if args.preserve_total_duration:
        # Pin total mel length to the source duration, ignoring any per-token durations.
        total_duration = src_token_len.long()
        duration = None
    else:
        total_duration = None

    # ---- Flow-matching decoder: tokens -> Mel ----
    # Note: SslVectorQuantizerJoint.synthesize signature mirrors `inference_joint.py`.
    speech_feat, speech_feat_len = quantizer.synthesize(
        quant_indices=tgt_token,
        feature_lengths=tgt_token_len,
        spk_embed=spk_embed,
        full_cfg=float(args.full_cfg),
        cond_cfg=float(args.cond_cfg),
        spk_cfg=float(args.spk_cfg),
        n_timesteps=int(args.n_timesteps),
        total_duration=total_duration,
        use_source_duration=(duration is not None),
    )

    # ---- HiFT vocoder: Mel -> waveform ----
    speech_feat_for_hift = speech_feat.transpose(1, 2)  # (1, D, T)
    hift_cache_source = torch.zeros(1, 1, 0, device=device)
    speech, _ = hift.inference(
        speech_feat=speech_feat_for_hift, cache_source=hift_cache_source
    )

    # ---- Save ----
    out_path = Path(args.output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), speech.squeeze(0).cpu().numpy(), sample_rate)
    print(f"[TokAN] Wrote audio: {out_path}")

    if args.mel_path:
        mel_path = Path(args.mel_path)
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(speech_feat.cpu(), mel_path)
        print(f"[TokAN] Wrote Mel tensor: {mel_path}")


if __name__ == "__main__":
    run_inference(parse_args())
