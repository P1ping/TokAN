import os
import logging
import pathlib
import argparse
from tqdm import tqdm
from typing import List
from contextlib import nullcontext

import numpy as np

import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class

from kaldiio import WriteHelper

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
ACCENT_MODEL_TAGS = ["ecapa", "w2v2"]


def load_audio(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        speech = torchaudio.functional.resample(speech, sample_rate, target_sr)
    return speech.squeeze(0)


def load_accent_model(model_tag, device):
    if model_tag == "w2v2":
        model = foreign_class(
            source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device},
        )
    elif model_tag == "ecapa":
        model = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="pretrained_models/accent-id-commonaccent_ecapa",
            run_opts={"device": device},
        )
    else:
        raise ValueError(f"Unknown model tag {model_tag}")

    return model


def extract_accent_feature(waveform, model, temperature=1.0):
    batch = waveform.unsqueeze(0).to(model.device)
    rel_length = torch.tensor([1.0])
    embeds = model.encode_batch(batch, rel_length)
    proj = model.mods.classifier if hasattr(model.mods, "classifier") else model.mods.output_mlp
    logits = proj(embeds).squeeze(1)
    probs = F.softmax(logits / temperature, dim=-1)
    return embeds.data.squeeze(), logits.data[0], probs.data[0]


@torch.inference_mode()
def recognize_accent(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_accent_model(args.model_tag, device=device)
    model.eval()

    shard_scp_path, shard_ark_path = worker_shard_path(args.output, args.local_rank)
    shard_ark_path.parent.mkdir(parents=True, exist_ok=True)
    ark_writer = WriteHelper(f"ark,scp:{shard_ark_path},{shard_scp_path}")

    # Open and process the input manifest
    with open(args.manifest, "r", encoding="utf-8") as f_in:
        header_line = next(f_in).strip()
        header = header_line.split("\t")

        # Get column indices
        utt_id_idx = header.index("id")
        src_wav_idx = header.index("src_audio")

        # Filter rows for this worker in distributed setting
        rows = []
        for line in f_in:
            row = line.strip().split("\t")
            rows.append(row)

        worker_rows = rows[args.local_rank :: args.world_size]

        for row in tqdm(worker_rows, desc=f"Worker {args.local_rank} processing"):
            utt_id = row[utt_id_idx]
            wav_path = row[src_wav_idx]
            try:
                waveform = load_audio(wav_path, SAMPLE_RATE)
                embed, logit, prob = extract_accent_feature(waveform, model, temperature=args.temperature)
                if args.output_type == "embed":
                    embed = F.layer_norm(embed, embed.shape)
                    accent_embed = embed.cpu().numpy()
                elif args.output_type == "logit":
                    logit = F.layer_norm(logit, logit.shape)
                    accent_embed = logit.cpu().numpy()
                elif args.output_type == "prob":
                    accent_embed = prob.cpu().numpy()
                else:
                    raise ValueError(f"Unknown output type {args.output_type}")
            except Exception as e:
                logger.error(f"Error processing wav {wav_path}: {e}")
                accent_embed = np.empty(0)

            ark_writer(utt_id, accent_embed)

    ark_writer.close()


def worker_shard_path(fname, worker_id) -> List[pathlib.Path]:
    base_path = str(pathlib.Path(fname + f".partial_{worker_id}").absolute())
    scp_path = base_path + ".scp"
    arc_path = base_path + ".ark"
    return pathlib.Path(scp_path), pathlib.Path(arc_path)


def merge_files(full_output, n_workers):
    with open(f"{full_output}.scp", "w") as full:
        for worker_id in tqdm(range(n_workers)):
            partial_path, _ = worker_shard_path(full_output, worker_id)
            partial_scp = open(partial_path, "r")
            for line in partial_scp:
                full.write(line)
            partial_path.unlink()


def write_manifest(input_manifest_path, accent_scp_path, output_manifest_path):
    # Load scp file into dictionary
    embed_dict = {}
    with open(accent_scp_path, "r", encoding="utf-8") as f_scp:
        for line in f_scp:
            parts = line.strip().split(" ")
            utt_id, embed_path = parts
            embed_dict[utt_id] = embed_path

    # Process original manifest and add accent embed paths
    with open(input_manifest_path, "r", encoding="utf-8") as f_in, open(
        output_manifest_path, "w", encoding="utf-8"
    ) as f_out:
        # Process header
        header_line = next(f_in).strip()
        header = header_line.split("\t")
        utt_id_idx = header.index("id")
        new_header = header + ["src_embed"]
        f_out.write("\t".join(new_header) + "\n")

        # Process data rows
        for line in f_in:
            row = line.strip().split("\t")
            utt_id = row[utt_id_idx]
            accent_embed_path = embed_dict.get(utt_id, "")
            new_row = row + [accent_embed_path]
            f_out.write("\t".join(new_row) + "\n")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_tag",
        default="w2v2",
        choices=ACCENT_MODEL_TAGS,
        help="Model tag to use, either ecapa, w2v2, or whisper model sizes",
    )
    parser.add_argument(
        "--output_type",
        default="embed",
        choices=["embed", "logit", "prob"],
        help="Output type, either embedding, logits or probabilities",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature for softmax, useful for extracting probabilities",
    )

    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest file")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--output_manifest",
        required=True,
        help="Path to the output manifest file with accent embeddings",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = get_parser()
    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

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
        recognize_accent(args)

    # Wait for all workers to finish
    if args.world_size > 1:
        torch.distributed.barrier()

    # Merge output files if leader
    if args.local_rank == 0:
        merge_files(args.output, args.world_size)
        write_manifest(
            input_manifest_path=args.manifest,
            accent_scp_path=f"{args.output}.scp",
            output_manifest_path=args.output_manifest,
        )
        logger.info(f"Written manifest with accent embeddings to: {args.output_manifest}")
