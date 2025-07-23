# TokAN: Accent Normalization Using Self-Supervised Discrete Tokens

This repository contains the official implementation and pretrained models for the Interspeech 2025 paper: **"Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data"**.

## Project Status: Inference Code Release

This repository currently provides the code and pretrained models required to run **inference** with the TokAN system.

The project is under development, and we are actively working to identify and fix potential bugs.

The code for **training** the models from scratch will be cleaned up and released at a later date.


## Installation

We strongly recommend using a `conda` or `venv` virtual environment with Python 3.9+. The installation process is sensitive to the order of operations. Please follow these steps for installation.

### Step 1: Clone Repository and Setup Environment

First, clone the repository, making sure to initialize the `fairseq` submodule. Then, create and activate your virtual environment.

```bash
# Clone the repository and all its submodules
git clone --recurse-submodules https://github.com/P1ping/TokAN.git
cd TokAN

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
# Or you can create a new environment via conda
```

### Step 2: Install PyTorch

The code has been run with torch versions `2.0.1` and `2.5.1`. So, we recommend a torch version in between, while other versions are probably compatible as well.

> For other CUDA versions or CPU-only installation, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command.

### Step 3: Install the Custom Fairseq Dependency

The `token-to-token` component of TokAN requires a specific version of Fairseq, which is included in the `third_party` directory. This **must be installed before** the other requirements to resolve a dependency conflict with `hydra-core`.
This **should be installed before** the the main project. Fairseq requires a lower version of `hydra-core`, which should be upgraded during installation of the main project.

```bash
# Navigate to the submodule directory
cd third_party/fairseq

# Install this specific version of fairseq in editable mode
pip install -e .

# Return to the project root directory
cd ../..
```

### Step 4: Install Project Dependencies and TokAN

Now that the critical dependencies are in place, you can install the remaining packages from `requirements.txt` and then install the TokAN source code itself.

```bash
# Install all other required packages
pip install -r requirements.txt

# Install the project's core source code in editable mode
pip install -e .
```

After these steps, the environment is fully configured to run inference.

## Running Inference

```bash
python inference.py --input_path /path/to/input.wav --output_path /path/to/output.wav --download_models
```

This will automatically download all required models and perform accent conversion on your audio file. Specifically, it will split the input audio (when it is long) into chunks and convert them one by one. The chunking is based on `silero_vad`, please check the arguments for detailed configurations. When `--preserve_duration` is given, the script will select the model with flow-matching-based duration prediction for total duration preservation.

## Acknowledgements

This project leverages several excellent open-source projects:

- [Fairseq](https://github.com/facebookresearch/fairseq): For training the accent-removal model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS): For text-to-speech synthesis to generate synthetic data
- [textlesslib](https://github.com/facebookresearch/textlesslib): For extract HuBERT tokens
- [BigVGAN](https://github.com/NVIDIA/BigVGAN): For high-quality neural vocoding

We express our gratitude to the authors and contributors of these projects for making their work available to the community.

For detailed information about our modifications to these codebases, please see the [Notes](tokan/README.md).

## Citation

If you find our work useful in your research, please cite our paper:

```bibtex
@inproceedings{bai2025accent,
  title={Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data},
  author={Bai, Qibing and Inoue, Sho and Wang, Shuai and Jiang, Zhongjie and Wang, Yannan and Li, Haizhou},
  booktitle={Proceedings of Interspeech},
  year={2025}
}