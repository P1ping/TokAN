# TokAN: Accent Normalization Using Self-Supervised Tokens

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![Demo](https://img.shields.io/badge/Demo-Samples-brightgreen.svg)](https://p1ping.github.io/TokAN-Samples)

Official open-source implementation of **"TokAN: Accent Normalization Using
Self-Supervised Tokens"**. TokAN performs accent normalization through an
autoregressive language model (ARLM) over self-supervised speech tokens: it
takes a source speech waveform and a reference speaker waveform, and emits a
new waveform that preserves the linguistic content of the source while
rendering it with the voice / accent profile implied by the reference.
This repository is currently inference-only.

- 📄 **Paper:** https://arxiv.org/abs/TODO *(placeholder — link to be updated)*
- 🔊 **Demo page:** https://p1ping.github.io/TokAN-Samples

> [!NOTE]
> TokAN is the journal extension of our earlier conference paper. The original
> implementation lives in the legacy repository:
> [P1ping/TokAN-Legacy](https://github.com/P1ping/TokAN-Legacy).

## Pipeline

```
src wav  ──► WavLM-Large(L22) + VQ ──► source token ids
source token ids ──► ARLM (TransformerLM) ──► target token ids
target token ids (deduplicated) ──► flow-matching decoder ──► Mel
Mel ──► HiFT vocoder ──► output waveform

reference wav ──► Resemblyzer ──► 256-d speaker embedding
```

The same `SslVectorQuantizerJoint` module is used twice: once to **quantize**
the source waveform into discrete tokens, and once to **synthesize** Mel
frames from the LM-emitted target tokens conditioned on the speaker
embedding.

## Inference Quick Start

```bash
pip install -r requirements.txt

python infer_wav.py \
    --source_wav   path/to/source.wav \
    --reference_wav path/to/reference.wav \
    --output_wav   outputs/result.wav
```

Checkpoints are auto-downloaded from Hugging Face on first use.

### Useful flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--reference_wav` | `source_wav` | Speaker / timbre reference. |
| `--beam_size` | `10` | ARLM beam-search width. Set `0` for sampling. |
| `--n_timesteps` | `32` | Flow-matching steps for the Mel decoder. |
| `--full_cfg` / `--cond_cfg` / `--spk_cfg` | `0.0 / 1.0 / 1.0` | Classifier-free guidance scales. |
| `--preserve_total_duration` | off | Force the output length to match the source. |
| `--use_lm_duration` | off | Use the LM-emitted token repeat counts as durations (otherwise the decoder's duration predictor is used). |
| `--device` | `auto` | `cpu` / `cuda` / `auto`. |
| `--mel_path` | unset | If given, also dump the predicted Mel as `.pt`. |

### Local checkpoints

To skip the Hugging Face download, pass local paths:

```bash
python infer_wav.py \
    --source_wav    path/to/source.wav \
    --output_wav    outputs/result.wav \
    --model_config  configs/tokan.model-only.yaml \
    --lm_checkpoint        path/to/arlm.pt \
    --quantizer_checkpoint path/to/quantizer.pt \
    --hift_checkpoint      path/to/hift.pt
```

## Repository layout

```
tokan/
├── infer_wav.py                  # user-facing inference entry point
├── configs/
│   └── tokan.model-only.yaml     # HyperPyYAML model definitions (lm / quantizer / hift)
├── requirements.txt
└── tokan/                        # python package
    ├── model/
    │   ├── arlm/                 # autoregressive speech LM (TransformerLM, RotaryEncoder)
    │   ├── ssl_quant/vq_joint.py # joint SSL VQ + flow-matching decoder
    │   ├── ssl_quant/dit_feat_flow.py
    │   ├── ssl_wrapper/wavlm_extractor.py
    │   ├── flow/                 # length regulator + duration predictor
    │   ├── dit/                  # DiT speech encoder + building blocks
    │   ├── hift/                 # HiFT vocoder
    │   ├── conv/                 # ResBlock1d, PostNet
    │   └── rope.py               # RoPE attention
    ├── transformer/              # ESPnet-style Conformer / Transformer primitives
    └── utils/                    # masks, RAS sampling, class registries
```

## Acknowledgements

- **CosyVoice / CosyVoice2**: HiFT vocoder architecture & pretrained weights;
  several Conformer building blocks borrowed via CosyVoice.
- **ESPnet**: Conformer / Transformer attention, embedding, and subsampling
  modules.
- **HuggingFace Transformers + Microsoft WavLM-Large**: SSL encoder used for
  source token extraction.
- **Resemblyzer**: speaker embedder used to condition the flow-matching
  decoder.
- **vector-quantize-pytorch**: VQ codebook implementation.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{tokan,
  title   = {TokAN: Accent Normalization Using Self-Supervised Tokens},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```
