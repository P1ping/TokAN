"""Interactive Gradio demo for TokAN wav-to-wav inference.

Run:

    python demo.py
    # or with explicit checkpoints / sharing:
    python demo.py --lm_checkpoint path/to/arlm.pt \
                   --quantizer_checkpoint path/to/quantizer.pt \
                   --hift_checkpoint path/to/hift.pt \
                   --share

If checkpoints are not provided, they are auto-downloaded from Hugging Face
(`Piping/TokAN`) via the same logic as ``infer_wav.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
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
# CLI for the demo launcher
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradio demo for TokAN wav-to-wav inference."
    )

    # Local checkpoints (optional; auto-downloaded if omitted)
    parser.add_argument("--model_config", default=None, help="Local config yaml path.")
    parser.add_argument("--lm_checkpoint", default=None, help="Local ARLM checkpoint (.pt).")
    parser.add_argument(
        "--quantizer_checkpoint", default=None,
        help="Local joint quantizer + flow decoder checkpoint (.pt).",
    )
    parser.add_argument("--hift_checkpoint", default=None, help="Local HiFT checkpoint (.pt).")

    # HF download
    parser.add_argument("--hf_repo_id", default="Piping/TokAN")
    parser.add_argument("--hf_subfolder", default="checkpoints")
    parser.add_argument("--model_tag", default="default")
    parser.add_argument("--lm_filename", default="arlm.pt")
    parser.add_argument("--quantizer_filename", default="quantizer.pt")
    parser.add_argument("--hift_filename", default="hift.pt")
    parser.add_argument("--config_filename", default=DEFAULT_CONFIG_FILENAME)
    parser.add_argument("--hf_revision", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--cache_dir", default="checkpoints")

    # Runtime
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--share", action="store_true", help="Expose a public Gradio link.")
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hf_download(*, repo_id, subfolder, model_tag, filename, cache_dir, revision, token):
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
    local_path: Optional[str], *, repo_id, subfolder, model_tag, filename,
    cache_dir, revision, token,
) -> str:
    if local_path is not None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")
        return local_path
    return _hf_download(
        repo_id=repo_id, subfolder=subfolder, model_tag=model_tag,
        filename=filename, cache_dir=cache_dir, revision=revision, token=token,
    )


def _audio_input_to_tensor(audio_input, target_sr: int) -> torch.Tensor:
    """Accept either a Gradio `(sr, np.ndarray)` tuple or a filepath; return mono float32."""
    if audio_input is None:
        raise gr.Error("Please provide a source waveform.")

    if isinstance(audio_input, str):
        wav, sr = torchaudio.load(audio_input)
        wav = wav.mean(dim=0)
    else:
        sr, data = audio_input
        if data is None or len(data) == 0:
            raise gr.Error("Empty audio input.")
        if data.ndim == 2:
            data = data.mean(axis=-1)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        wav = torch.from_numpy(data.astype(np.float32))

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), orig_freq=sr, new_freq=target_sr
        ).squeeze(0)
    return wav


def _deduplicate(tokens):
    """Run-length-encode token list -> (unique tokens, durations)."""
    if not tokens:
        return [], []
    out_tokens, out_dur = [], []
    prev, count = tokens[0], 1
    for t in tokens[1:]:
        if t == prev:
            count += 1
        else:
            out_tokens.append(prev)
            out_dur.append(count)
            prev, count = t, 1
    out_tokens.append(prev)
    out_dur.append(count)
    return out_tokens, out_dur


# ---------------------------------------------------------------------------
# Inference engine (eager-loaded at startup)
# ---------------------------------------------------------------------------
class TokANInferenceEngine:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.device = device

        lm_ckpt = _resolve_or_download(
            args.lm_checkpoint, repo_id=args.hf_repo_id, subfolder=args.hf_subfolder,
            model_tag=args.model_tag, filename=args.lm_filename, cache_dir=args.cache_dir,
            revision=args.hf_revision, token=args.hf_token,
        )
        quantizer_ckpt = _resolve_or_download(
            args.quantizer_checkpoint, repo_id=args.hf_repo_id, subfolder=args.hf_subfolder,
            model_tag=args.model_tag, filename=args.quantizer_filename, cache_dir=args.cache_dir,
            revision=args.hf_revision, token=args.hf_token,
        )
        hift_ckpt = _resolve_or_download(
            args.hift_checkpoint, repo_id=args.hf_repo_id, subfolder=args.hf_subfolder,
            model_tag=args.model_tag, filename=args.hift_filename, cache_dir=args.cache_dir,
            revision=args.hf_revision, token=args.hf_token,
        )

        if args.model_config is not None:
            if not os.path.isfile(args.model_config):
                raise FileNotFoundError(f"Config file not found: {args.model_config}")
            config_path = args.model_config
        else:
            config_path = _hf_download(
                repo_id=args.hf_repo_id, subfolder=args.hf_subfolder, model_tag=args.model_tag,
                filename=args.config_filename, cache_dir=args.cache_dir,
                revision=args.hf_revision, token=args.hf_token,
            )

        print(f"[TokAN] Loading config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            configs = load_hyperpyyaml(f)

        print("[TokAN] Loading ARLM ...")
        self.lm = configs["lm"].to(device).eval()
        lm_state = torch.load(lm_ckpt, map_location="cpu")
        for key in ("epoch", "step"):
            lm_state.pop(key, None)
        self.lm.load_state_dict(lm_state, strict=True)
        for p in self.lm.parameters():
            p.requires_grad = False

        print("[TokAN] Loading joint quantizer + flow decoder ...")
        self.quantizer = configs["quantizer"].to(device).eval()
        q_state = torch.load(quantizer_ckpt, map_location="cpu")
        for key in ("epoch", "step"):
            q_state.pop(key, None)
        self.quantizer.load_state_dict(q_state, strict=True)
        for p in self.quantizer.parameters():
            p.requires_grad = False

        print("[TokAN] Loading HiFT vocoder ...")
        self.hift = configs["hift"].to(device).eval()
        self.hift.load_state_dict(torch.load(hift_ckpt, map_location="cpu"), strict=True)
        for p in self.hift.parameters():
            p.requires_grad = False

        self.sample_rate = int(configs["sample_rate"])

        print("[TokAN] Loading speaker encoder (Resemblyzer) ...")
        from resemblyzer import VoiceEncoder
        self.spk_encoder = VoiceEncoder(device="cuda" if device.type == "cuda" else "cpu")
        self.spk_encoder.eval()

        print("[TokAN] All models loaded.")

    @torch.inference_mode()
    def _extract_spk_embedding(self, ref_wav_16k: torch.Tensor) -> torch.Tensor:
        from resemblyzer import preprocess_wav
        # Resemblyzer takes a numpy waveform at 16 kHz.
        preprocessed = preprocess_wav(ref_wav_16k.cpu().numpy())
        emb = self.spk_encoder.embed_utterance(preprocessed)
        emb_t = torch.from_numpy(np.asarray(emb, dtype=np.float32)).unsqueeze(0)
        emb_t = F.normalize(emb_t, p=2, dim=-1)
        return emb_t.to(self.device)

    @torch.inference_mode()
    def synthesize(
        self,
        source_audio,
        reference_audio,
        beam_size: int,
        duration_mode: str,
        duration_scale: float,
        n_timesteps: int,
        full_cfg: float,
        cond_cfg: float,
        spk_cfg: float,
    ):
        """Run the full pipeline. Returns ((sample_rate, np.float32 waveform), info_text)."""
        # ---- Source wav (16 kHz mono for the WavLM extractor) ----
        src_wav = _audio_input_to_tensor(source_audio, target_sr=16000).to(self.device)
        src_wav = src_wav.unsqueeze(0)  # (1, T_wav)
        src_wav_len = torch.tensor([src_wav.shape[1]], dtype=torch.long, device=self.device)

        # ---- Reference wav (default to source) ----
        ref_input = reference_audio if reference_audio is not None else source_audio
        ref_wav_16k = _audio_input_to_tensor(ref_input, target_sr=16000)
        spk_embed = self._extract_spk_embedding(ref_wav_16k)

        # ---- WavLM + VQ -> source tokens ----
        _, src_token, src_token_len = self.quantizer.quantize(src_wav, src_wav_len)
        n_src_tokens = int(src_token_len.item())

        # ---- ARLM -> target tokens ----
        token_list = []
        for tok in self.lm.inference(
            src_tokens=src_token,
            src_token_len=src_token_len,
            beam_size=int(beam_size),
        ):
            token_list.append(int(tok))
        if not token_list:
            raise gr.Error("ARLM produced an empty token sequence; try a different source.")

        # ---- Tokens -> deduplicated tokens (the decoder expects dedup tokens) ----
        if self.quantizer.flow.duration_predictor is not None:
            tgt_token_list, _lm_durations = _deduplicate(token_list)
            tgt_token = torch.tensor(
                tgt_token_list, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            tgt_token_len = torch.tensor(
                [tgt_token.shape[1]], dtype=torch.long, device=self.device
            )
        else:
            tgt_token = torch.tensor(
                token_list, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            tgt_token_len = torch.tensor(
                [tgt_token.shape[1]], dtype=torch.long, device=self.device
            )

        # ---- Duration handling ----
        # duration_mode in {"predicted", "scale"}.
        #   predicted: the flow decoder's duration predictor decides; total_duration=None.
        #   scale:     total_duration = round(n_src_tokens * duration_scale).
        #              A scale of 1.0 preserves source total duration.
        if duration_mode == "predicted":
            total_duration = None
        else:
            scaled = max(1, int(round(n_src_tokens * float(duration_scale))))
            total_duration = torch.tensor([scaled], dtype=torch.long, device=self.device)

        # ---- Flow matching -> Mel ----
        speech_feat, _ = self.quantizer.synthesize(
            quant_indices=tgt_token,
            feature_lengths=tgt_token_len,
            spk_embed=spk_embed,
            full_cfg=float(full_cfg),
            cond_cfg=float(cond_cfg),
            spk_cfg=float(spk_cfg),
            n_timesteps=int(n_timesteps),
            total_duration=total_duration,
            use_source_duration=False,
        )

        # ---- HiFT -> waveform ----
        hift_in = speech_feat.transpose(1, 2)  # (1, D, T)
        cache_source = torch.zeros(1, 1, 0, device=self.device)
        speech, _ = self.hift.inference(speech_feat=hift_in, cache_source=cache_source)
        speech_np = speech.squeeze(0).cpu().numpy().astype(np.float32)

        info = (
            f"src tokens: {n_src_tokens} | "
            f"emitted: {len(token_list)} | "
            f"dedup: {tgt_token.shape[1]} | "
            f"mel frames: {speech_feat.shape[1]} | "
            f"audio: {speech_np.shape[-1] / self.sample_rate:.2f}s"
        )
        return (self.sample_rate, speech_np), info


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
DESCRIPTION = """
# TokAN — Token-based Speech Conversion

Upload (or record) a **source** waveform and an optional **reference** speaker
waveform. TokAN will:

1. Quantize the source via WavLM + VQ → discrete source tokens.
2. Convert source tokens → target tokens with the autoregressive ARLM.
3. Decode target tokens → Mel with a flow-matching speech decoder, conditioned
   on the reference speaker embedding (Resemblyzer).
4. Vocode Mel → waveform with HiFT.

If no reference is given, the source itself is used as the speaker reference
(timbre-preserving conversion).
"""


def build_demo(engine: TokANInferenceEngine) -> gr.Blocks:

    def run(
        source_audio,
        reference_audio,
        beam_size,
        duration_mode,
        duration_scale,
        n_timesteps,
        full_cfg,
        cond_cfg,
        spk_cfg,
    ):
        try:
            audio_out, info = engine.synthesize(
                source_audio=source_audio,
                reference_audio=reference_audio,
                beam_size=int(beam_size),
                duration_mode=duration_mode,
                duration_scale=float(duration_scale),
                n_timesteps=int(n_timesteps),
                full_cfg=float(full_cfg),
                cond_cfg=float(cond_cfg),
                spk_cfg=float(spk_cfg),
            )
            return audio_out, info
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(f"Inference failed: {exc}")

    with gr.Blocks(title="TokAN Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                source_audio = gr.Audio(
                    label="Source speech (content)",
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                reference_audio = gr.Audio(
                    label="Reference speaker (timbre) — optional, defaults to source",
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                run_button = gr.Button("Convert", variant="primary")

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Converted speech", type="numpy", interactive=False
                )
                info_box = gr.Textbox(label="Run info", interactive=False, lines=2)

        with gr.Accordion("Advanced inference parameters", open=False):
            with gr.Row():
                beam_size = gr.Slider(
                    minimum=0, maximum=20, step=1, value=10,
                    label="Beam size (0 = sampling)",
                )
                n_timesteps = gr.Slider(
                    minimum=4, maximum=64, step=1, value=32,
                    label="Flow-matching steps",
                )

            gr.Markdown(
                "**Duration control.** *predicted*: the flow decoder's duration "
                "predictor decides the length. *scale*: total output frames = "
                "`source_frames × duration scale`; 1.0 preserves the source "
                "duration, >1.0 lengthens, <1.0 shortens."
            )
            with gr.Row():
                duration_mode = gr.Radio(
                    choices=["predicted", "scale"],
                    value="predicted",
                    label="Duration mode",
                )
                duration_scale = gr.Slider(
                    minimum=0.5, maximum=2.0, step=0.05, value=1.0,
                    label="Duration scale (used when mode = scale)",
                )

            gr.Markdown(
                "**Classifier-free guidance (CFG)** for the flow-matching decoder. "
                "`cond_cfg` controls how strongly the target tokens steer the Mel; "
                "`spk_cfg` controls how strongly the reference speaker steers it; "
                "`full_cfg` is the joint (null-branch) guidance scale."
            )
            with gr.Row():
                full_cfg = gr.Slider(
                    minimum=0.0, maximum=3.0, step=0.05, value=0.0, label="full_cfg"
                )
                cond_cfg = gr.Slider(
                    minimum=0.0, maximum=3.0, step=0.05, value=1.0, label="cond_cfg"
                )
                spk_cfg = gr.Slider(
                    minimum=0.0, maximum=3.0, step=0.05, value=1.0, label="spk_cfg"
                )

        run_button.click(
            fn=run,
            inputs=[
                source_audio, reference_audio,
                beam_size, duration_mode, duration_scale,
                n_timesteps, full_cfg, cond_cfg, spk_cfg,
            ],
            outputs=[output_audio, info_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = _resolve_device(args.device)
    print(f"[TokAN] Using device: {device}")

    engine = TokANInferenceEngine(args, device=device)
    demo = build_demo(engine)
    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
