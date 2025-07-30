import os
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import torch
import torchaudio

import jiwer
from transformers import pipeline
from whisper.normalizers.english import EnglishTextNormalizer

from tokan.data.l2arctic import SPEAKER_TO_ACCENT, UNSEEN_SPEAKERS


ASR_HF_TAG = "facebook/s2t-medium-librispeech-asr"


def load_manifest(manifest_path, tag_to_recognize):
    with open(manifest_path) as f:
        header = f.readline().strip()
        keys = header.split("\t")
        assert "id" in keys, "The manifest file must contain 'id' field"
        assert tag_to_recognize in keys, f"The manifest file must contain '{tag_to_recognize}' field"
        assert "text" in keys, "The manifest file must contain 'text' field"

        manifest = {k: [] for k in keys}

        for line in f:
            for k, v in zip(keys, line.strip().split("\t")):
                manifest[k].append(v)

    return manifest


def load_audio(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        if sample_rate < target_sr:
            print("Warning: wav sample rate {} is less than target sample rate {}".format(sample_rate, target_sr))
        speech = torchaudio.functional.resample(speech, sample_rate, target_sr)
    return speech


def worker_shard_path(fname, worker_id) -> Path:
    return Path(fname).with_suffix(f".partial_{worker_id}")


def merge_files(full_output, n_workers):
    manifest = []
    for worker_id in range(n_workers):
        partial_path = worker_shard_path(full_output, worker_id)
        with open(partial_path, "r") as f:
            header_line = f.readline()
            for line in f:
                manifest.append(line)
        partial_path.unlink()
    return [header_line] + manifest


def load_merged_results(output_transcript):
    with open(output_transcript) as f:
        header = f.readline().strip()
        keys = header.split("\t")
        assert keys[0] == "utt_id"
        assert keys[1] == "ground_truth"
        assert keys[2] == "recognized"

        manifest = {k: [] for k in keys}
        for line in f:
            for k, v in zip(keys, line.strip().split("\t")):
                manifest[k].append(v)

    return manifest["utt_id"], manifest["ground_truth"], manifest["recognized"]


def recognize_list(args):
    manifest = load_manifest(args.manifest_path, args.tag_to_recognize)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pipeline("automatic-speech-recognition", model=ASR_HF_TAG, device=device)

    # Recognize audios
    synth_transcription_list = []
    gt_transcription_list = []

    for idx in tqdm(range(args.local_rank, len(manifest["id"]), args.world_size)):
        synth_path = manifest[args.tag_to_recognize][idx]
        text = manifest["text"][idx]

        synth_transcription = model(synth_path)["text"]

        synth_transcription_list.append(synth_transcription)
        gt_transcription_list.append(text)

    local_output_path = worker_shard_path(args.output_transcript, args.local_rank)
    with open(local_output_path, "w") as f:
        f.write(f"utt_id\tground_truth\trecognized\n")
        for utt_id, gt, synth in zip(
            manifest["id"][args.local_rank :: args.world_size],
            gt_transcription_list,
            synth_transcription_list,
        ):
            f.write(f"{utt_id}\t{gt}\t{synth}\n")
    print(f"Saved reconized transcripts to {local_output_path}")


def compute_wer(utt_list, gt_transcription_list, synth_transcription_list):
    normalizer = EnglishTextNormalizer()
    synth_transcription_list = [normalizer(t) for t in synth_transcription_list]
    gt_transcription_list = [normalizer(t) for t in gt_transcription_list]

    # Overall WER
    overall_wer = jiwer.wer(gt_transcription_list, synth_transcription_list)

    # Compute WERs for different subsets by concatenating their transcriptions
    subset2texts = {"seen": {"gt": [], "synth": []}, "unseen": {"gt": [], "synth": []}}
    speaker2texts = defaultdict(lambda: {"gt": [], "synth": []})
    accent2texts = defaultdict(lambda: {"gt": [], "synth": []})

    for utt_id, gt, synth in zip(utt_list, gt_transcription_list, synth_transcription_list):
        speaker = utt_id.split("_")[0]
        accent = SPEAKER_TO_ACCENT[speaker]

        # Sort into seen/unseen
        subset = "unseen" if speaker in UNSEEN_SPEAKERS else "seen"
        subset2texts[subset]["gt"].append(gt)
        subset2texts[subset]["synth"].append(synth)

        # Group by speaker
        speaker2texts[speaker]["gt"].append(gt)
        speaker2texts[speaker]["synth"].append(synth)

        # Group by accent
        accent2texts[accent]["gt"].append(gt)
        accent2texts[accent]["synth"].append(synth)

    # Compute WERs for each group
    subset2wer = {subset: jiwer.wer(texts["gt"], texts["synth"]) for subset, texts in subset2texts.items()}

    speaker2wer = {speaker: jiwer.wer(texts["gt"], texts["synth"]) for speaker, texts in speaker2texts.items()}

    accent2wers = {accent: jiwer.wer(texts["gt"], texts["synth"]) for accent, texts in accent2texts.items()}

    return overall_wer, subset2wer, speaker2wer, accent2wers


def get_parser():
    parser = argparse.ArgumentParser(description="Speech recognition evaluation script")
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to the metadata file",
    )
    parser.add_argument(
        "--output_transcript",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_results",
        type=str,
        required=True,
    )
    parser.add_argument("--tag_to_recognize", type=str, default="gen_audio")
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

    os.makedirs(os.path.dirname(args.output_transcript), exist_ok=True)

    with context:
        recognize_list(args)

    if args.world_size > 1:
        torch.distributed.barrier()

    if args.local_rank == 0:
        output_manifest = merge_files(args.output_transcript, args.world_size)
        with open(args.output_transcript, "w") as f:
            f.writelines(output_manifest)

        utt_list, gt_transcription_list, synth_transcription_list = load_merged_results(args.output_transcript)
        overall_wer, subset2wer, speaker2wer, accent2wers = compute_wer(
            utt_list, gt_transcription_list, synth_transcription_list
        )

        os.makedirs(os.path.dirname(args.output_results), exist_ok=True)
        with open(args.output_results, "w") as f:
            f.write(f"Overall WER: {overall_wer * 100:.2f}%\n")
            f.write("-----------Subset WERs-----------:\n")
            for sub, wer in subset2wer.items():
                f.write(f"{sub}: {wer * 100:.2f}%\n")
            f.write("-----------Speaker WERs-----------:\n")
            for spk, wer in speaker2wer.items():
                f.write(f"{spk}: {wer * 100:.2f}%\n")
            f.write("-----------Accent WERs-----------\n")
            for acc, wer in accent2wers.items():
                f.write(f"{acc}: {wer * 100:.2f}%\n")

        print(f"Saved WER results to {args.output_results}")
        print(f"Overall WER: {overall_wer * 100:.2f}%")
