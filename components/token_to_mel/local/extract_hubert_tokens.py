# Modified from https://github.com/facebookresearch/textlesslib/blob/ba33d669d8284b4f7bfe81e7384e83ab799fe384/tools/distributed_transcribe/transcribe.py

import os
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext

import torch
import torchaudio

from tokan.textless.speech_encoder import SpeechEncoder
from tokan.textless.kmeans_quantizer import KMeansQuantizer
from tokan.textless.hubert_feature_reader import HubertFeatureReader


logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000


def load_metadata(metadata_path, split="|"):
    metadata = []
    with open(metadata_path, "r") as f:
        for line in f:
            utt_id, wav_path, *res = line.strip().split(split)
            utt_id = os.path.splitext(os.path.basename(wav_path))[0]
            metadata.append((utt_id, wav_path, *res))
    return metadata


def load_audio(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        speech = torchaudio.functional.resample(speech, sample_rate, target_sr)
    return speech.squeeze(0)


def transcribe(args):
    # Setup encoders
    hubert = HubertFeatureReader(args.dense_model, layer=args.layer)
    kmeans = KMeansQuantizer(args.quantizer_model)
    speech_encoder = SpeechEncoder(
        hubert,
        kmeans,
        need_f0=False,
        deduplicate=False,
        padding=True,
    ).cuda()

    metadata = load_metadata(args.manifest, args.split)
    output_path = worker_shard_path(args.output_manifest, args.local_rank)

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        for i in tqdm(range(args.local_rank, len(metadata), args.world_size)):
            utt_id, wav_path, *res = metadata[i]
            try:
                src_waveform = load_audio(wav_path, SAMPLING_RATE)
                src_encoded = speech_encoder(src_waveform)
                src_tokens = [str(int(x)) for x in src_encoded["units"].tolist()]
                src_tokens_str = " ".join(src_tokens)
            except Exception as e:
                logger.error(f"Error processing source wav {wav_path}: {e}")
                src_tokens_str = ""
            line = args.split.join([utt_id, wav_path, *res, src_tokens_str]) + "\n"
            f_out.write(line)


def worker_shard_path(output, worker_id) -> Path:
    return Path(f"{output}.partial_{worker_id}")


def merge_files(output, n_workers):
    with open(output, "w") as full:
        for worker_id in tqdm(range(n_workers)):
            partial_path = worker_shard_path(output, worker_id)
            partial_scp = open(partial_path, "r")
            for line in partial_scp:
                full.write(line)
            partial_path.unlink()


def get_parser():
    parser = argparse.ArgumentParser(description="Extract discrete tokens from speech")
    parser.add_argument("--dense_model", default="hubert-base-ls960", help="Dense model to be used")
    parser.add_argument("--layer", type=int, default=6, help="Layer to be used for feature extraction")
    parser.add_argument("--quantizer_model", default="kmeans", help="Quantizer model to be used")
    parser.add_argument(
        "--manifest", required=True, help="Path to the TSV manifest file with headers: id, src_audio, tgt_audio, text"
    )
    parser.add_argument("--output_manifest", required=True, help="Path to the output manifest file")
    parser.add_argument("--split", type=str, default="|", help="Delimiter for manifest file")

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

    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)

    with context:
        transcribe(args)

    # Wait for all workers to finish
    if args.world_size > 1:
        torch.distributed.barrier()

    # Merge output files if leader
    if args.local_rank == 0:
        merge_files(args.output_manifest, args.world_size)
