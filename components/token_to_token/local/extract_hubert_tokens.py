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
        deduplicate=args.deduplicate,
        padding=True,
    ).cuda()

    # Open output file for this worker
    output_path = worker_shard_path(args.output, args.local_rank)
    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        # Write header
        f_out.write("id\tsrc_audio\ttgt_audio\ttext\tsrc_tokens\ttgt_tokens\tsrc_n_tokens\ttgt_n_tokens\n")

        # Open and process the input manifest
        with open(args.manifest, "r", encoding="utf-8") as f_in:
            header_line = next(f_in).strip()
            header = header_line.split("\t")

            # Get column indices
            utt_id_idx = header.index("id")
            src_wav_idx = header.index("src_audio")
            tgt_wav_idx = None if args.use_src_as_tgt else header.index("tgt_audio")
            text_idx = header.index("text")

            # Filter rows for this worker in distributed setting
            rows = []
            for line in f_in:
                row = line.strip().split("\t")
                rows.append(row)

            worker_rows = rows[args.local_rank :: args.world_size]

            for row in tqdm(worker_rows, desc=f"Worker {args.local_rank} processing"):
                utt_id = row[utt_id_idx]
                src_wav_path = row[src_wav_idx]
                tgt_wav_path = src_wav_path if args.use_src_as_tgt else row[tgt_wav_idx]
                text = row[text_idx]

                # Process source wavefile
                try:
                    src_waveform = load_audio(src_wav_path, SAMPLING_RATE)
                    src_encoded = speech_encoder(src_waveform)
                    src_tokens = [str(int(x)) for x in src_encoded["units"].tolist()]
                    src_tokens_str = " ".join(src_tokens)
                    src_n_tokens = len(src_tokens)
                except Exception as e:
                    logger.error(f"Error processing source wav {src_wav_path}: {e}")
                    src_tokens_str = ""
                    src_n_tokens = 0

                # If using source tokens as target tokens
                if args.use_src_as_tgt:
                    tgt_tokens_str = src_tokens_str
                    tgt_n_tokens = src_n_tokens
                else:
                    # Process target wavefile
                    try:
                        tgt_waveform = load_audio(tgt_wav_path, SAMPLING_RATE)
                        tgt_encoded = speech_encoder(tgt_waveform)
                        tgt_tokens = [str(int(x)) for x in tgt_encoded["units"].tolist()]
                        tgt_tokens_str = " ".join(tgt_tokens)
                        tgt_n_tokens = len(tgt_tokens)
                    except Exception as e:
                        logger.error(f"Error processing target wav {tgt_wav_path}: {e}")
                        tgt_tokens_str = ""
                        tgt_n_tokens = 0

                # Write output row directly as TSV
                f_out.write(
                    f"{utt_id}\t{src_wav_path}\t{tgt_wav_path}\t{text}\t{src_tokens_str}\t{tgt_tokens_str}\t{src_n_tokens}\t{tgt_n_tokens}\n"
                )


def worker_shard_path(output, worker_id) -> Path:
    return Path(output).with_suffix(f".partial_{worker_id}.tsv")


def merge_files(output, n_workers):
    """Merge worker files into final output file"""
    output_path = Path(output)

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        # Write header first
        f_out.write("id\tsrc_audio\ttgt_audio\ttext\tsrc_tokens\ttgt_tokens\tsrc_n_tokens\ttgt_n_tokens\n")

        # Merge all worker files
        for worker_id in range(n_workers):
            worker_path = worker_shard_path(output, worker_id)
            with open(worker_path, "r", encoding="utf-8") as f_in:
                next(f_in)  # Skip header
                for line in f_in:
                    f_out.write(line)
            # Remove worker file
            worker_path.unlink()

    logger.info(f"Merged output saved to {output_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="Extract discrete tokens from speech")
    parser.add_argument("--dense_model", default="hubert-base-ls960", help="Dense model to be used")
    parser.add_argument("--layer", type=int, default=6, help="Layer to be used for feature extraction")
    parser.add_argument("--quantizer_model", default="kmeans", help="Quantizer model to be used")
    parser.add_argument(
        "--manifest", required=True, help="Path to the TSV manifest file with headers: id, src_audio, tgt_audio, text"
    )
    parser.add_argument("--output", required=True, help="Directory where token files will be saved")
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="If set, consecutive repeats of the same token are collapsed ('1 2 2 2 3' becomes '1 2 3')",
    )
    parser.add_argument(
        "--use_src_as_tgt",
        action="store_true",
        help="If set, use source tokens as target tokens instead of extracting from target wav",
    )

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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with context:
        transcribe(args)

    # Wait for all workers to finish
    if args.world_size > 1:
        torch.distributed.barrier()

    # Merge output files if leader
    if args.local_rank == 0:
        merge_files(args.output, args.world_size)
