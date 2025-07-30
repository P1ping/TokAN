import os
import torch
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import nullcontext

import soundfile as sf

import torch
import torch.distributed

from resemblyzer import VoiceEncoder, preprocess_wav

# Matcha imports
from tokan.matcha.models.matcha_tts import MatchaTTS
from tokan.matcha.text import sequence_to_text, text_to_sequence
from tokan.matcha.utils.utils import intersperse

# Vocoder imports
from tokan.utils.model_utils import load_bigvgan

# L2ARCTIC imports
from tokan.data.l2arctic import SEEN_SPEAKERS
from tokan.data.l2arctic import load_metadata, get_sentence_to_subset


def load_model(checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model


@torch.inference_mode()
def process_text(text, device):
    x = torch.tensor(intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0), dtype=torch.long, device=device)[
        None
    ]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


@torch.inference_mode()
def synthesise(text, spks, model, n_timesteps=32, temperature=0.667, length_scale=1.0):
    text_processed = process_text(text, model.device)
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        cond=spks,
        length_scale=length_scale,
    )
    return output


def save_to_folder(filename, audio, folder, sr):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    output_path = folder / f"{filename}.wav"
    sf.write(folder / f"{filename}.wav", audio, sr, "PCM_16")
    return str(output_path)


def worker_shard_path(directory, subset, worker_id) -> Path:
    return Path(directory).absolute() / f".{subset}.partial_{worker_id}.raw.tsv"


def merge_files(directory, subset, n_workers):
    output = Path(directory).absolute() / f"{subset}.raw.tsv"
    with open(output, "w") as full:
        full.write("id\tsrc_audio\ttgt_audio\ttext\n")
        for worker_id in range(n_workers):
            partial_path = worker_shard_path(directory, subset, worker_id)
            partial = open(partial_path, "r")
            _ = partial.readline()  # Skip header
            for line in partial:
                print(line.strip(), file=full)
            partial_path.unlink()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model = load_model(args.matcha_ckpt, device)
    count_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"
    print(f"Model loaded! Parameter count: {count_params(model)}")

    speaker_encoder = VoiceEncoder(device=device)
    vocoder = load_bigvgan(args.bigvgan_tag_or_ckpt, device)
    sampling_rate = vocoder.h["sampling_rate"]

    # Load dataset metadata
    metadata = load_metadata(args.data_dir)
    sid2subset = get_sentence_to_subset()

    manifests = {"train": [], "valid": [], "test": []}
    for i in tqdm(range(args.local_rank, len(metadata), args.world_size)):
        spk, sid, text, wav_path = metadata[i]
        utt_id = f"{spk}_{sid}"

        if not os.path.exists(wav_path):
            print(f"Skipped {utt_id} as the file is missing")
            continue

        output_wav_path = os.path.join(args.output_wav_dir, f"{utt_id}.wav")
        if not os.path.exists(output_wav_path):
            spk_wav = preprocess_wav(wav_path)
            spks = speaker_encoder.embed_utterance(spk_wav)
            spks = torch.FloatTensor(spks).unsqueeze(0).to(device)

            output = synthesise(text, spks, model)
            waveform = vocoder(output["mel"]).clamp(-1, 1).cpu().squeeze()
            output_wav_path = save_to_folder(utt_id, waveform, args.output_wav_dir, sampling_rate)

        item = [utt_id, wav_path, output_wav_path, text]
        sample_subset = sid2subset[sid]

        if sample_subset == "test":
            manifests["test"].append(item)
        elif spk in SEEN_SPEAKERS:
            manifests[sample_subset].append(item)
        else:
            print(f"Dropped sample {spk}'s {sid} in training/validation")

    for subset in ["train", "valid", "test"]:
        with open(worker_shard_path(args.output_manifest_dir, subset, args.local_rank), "w") as f:
            f.write("id\tsrc_audio\ttgt_audio\ttext\n")
            for item in manifests[subset]:
                f.write(f"{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n")


def get_parser():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--matcha_ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument(
        "--bigvgan_tag_or_ckpt", type=str, default="nvidia/bigvgan_22khz_80band", help="Tag for BigVGAN model"
    )
    parser.add_argument("--output_wav_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--output_manifest_dir", type=str, required=True, help="Path to the output manifest")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.distributed.init_process_group(backend="gloo", world_size=args.world_size, init_method="env://")
    n_gpus = torch.cuda.device_count()
    device_id = args.local_rank % n_gpus if n_gpus > 0 else None
    context = torch.cuda.device(device_id) if device_id is not None else nullcontext()

    print(f"WORLD_SIZE: {args.world_size}, LOCAL_RANK: {args.local_rank}")
    print(f"#GPUs: {n_gpus}, GPU ID: {device_id}")

    os.makedirs(args.output_wav_dir, exist_ok=True)
    os.makedirs(args.output_manifest_dir, exist_ok=True)

    with context:
        main(args)

    if args.world_size > 1:
        torch.distributed.barrier()

    if args.local_rank == 0:
        for subset in ["train", "valid", "test"]:
            merge_files(args.output_manifest_dir, subset, args.world_size)
        print(f"Generated manifests saved to {args.output_manifest_dir}")
