# Technical Notes: Code Adaptations in TokAN

This document provides details about the modifications and adaptations we've made to the open-source projects used in TokAN.

## Overview of Code Structure

The `tokan` directory contains several modules, some of which are adapted from existing open-source projects:

```
tokan/
├── bigvgan/                  # Adapted from NVIDIA's BigVGAN
├── fairseq_modules/          # fairseq-compatible modules
├── textless/                 # Adapted from Meta's textlesslib
├── yirga/                    # Adapted from Matcha-TTS
├── utils/                    # Utility functions and classes
└── └── model_utils.py        # Model loading and management utilities
```

## Adaptations from Open-Source Projects

### 1. Fairseq Adaptations

**Original Repository**: [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)

**Key Modifications**:
- Custom encoder/decoder architectures for token-to-token conversion
- Conditioning on accent embeddings

### 2. textlesslib Adaptations

**Original Repository**: [facebookresearch/textlesslib](https://github.com/facebookresearch/textlesslib)

**Key Modifications**:
- Padding audio before network processing.

### 3. Matcha-TTS Adaptations (in yirga/)

**Original Repository**: [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

**Key Modifications**:

- Injection of speaker embedding via AdaLN.

- Cosine time scheduling.

- Additional bracket and nasalization symbols in `text/symbols.py`

- Classifier-free guidance (CFG)

### 4. BigVGAN Adaptations

**Original Repository**: [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)

## Implementation Notes

### Model Integration

The various components have been integrated to create a seamless pipeline for accent conversion:
1. Content extraction using HuBERT and K-means from textlesslib
2. Token-to-token conversion using our custom fairseq-based models
3. Mel-spectrogram synthesis using the adapted Matcha-TTS components (named Yirga)
4. Waveform generation using the BigVGAN vocoder


## License Information

Each adapted component retains its original license:
- Fairseq: MIT License
- textlesslib: MIT License
- Matcha-TTS: MIT License
- BigVGAN: MIT License

Our modifications and original code in TokAN are released under MIT License.