import os
import argparse
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict

import numpy as np

import torch

from resemblyzer import VoiceEncoder, preprocess_wav

from tokan.data.l2arctic import SPEAKER_TO_ACCENT, UNSEEN_SPEAKERS


def load_manifest(manifest_path, tag_to_recognize, reference_tag):
    with open(manifest_path) as f:
        header = f.readline().strip()
        keys = header.split("\t")
        assert "id" in keys, "The manifest file must contain 'id' field"
        assert tag_to_recognize in keys, f"The manifest file must contain '{tag_to_recognize}' field"
        assert reference_tag in keys, f"The manifest file must contain '{reference_tag}' field"

        manifest = {k: [] for k in keys}

        for line in f:
            for k, v in zip(keys, line.strip().split("\t")):
                manifest[k].append(v)

    return manifest


@torch.inference_mode()
def compute_speaker_similarity(model, audio_path_1, audio_path_2):
    # Load audio
    audio_1 = preprocess_wav(audio_path_1)
    audio_2 = preprocess_wav(audio_path_2)

    # Compute embeddings
    emb_1 = model.embed_utterance(audio_1)
    emb_2 = model.embed_utterance(audio_2)

    emb_1 = emb_1 / np.linalg.norm(emb_1, axis=0, ord=2)
    emb_2 = emb_2 / np.linalg.norm(emb_1, axis=0, ord=2)

    # Compute similarity
    similarity = emb_1 @ emb_2.T

    return similarity.item()


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


def compute_similarity_list(args):
    manifest = load_manifest(args.manifest_path, args.tag_to_recognize, args.reference_tag)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VoiceEncoder(device=device)

    similarity_list = []
    for idx in tqdm(range(args.local_rank, len(manifest["id"]), args.world_size)):
        audio_path_1 = manifest[args.tag_to_recognize][idx]
        audio_path_2 = manifest[args.reference_tag][idx]
        similarity = compute_speaker_similarity(model, audio_path_1, audio_path_2)
        similarity_list.append(similarity)

    local_output_path = worker_shard_path(args.output_similarities, args.local_rank)
    with open(local_output_path, "w") as f:
        f.write(f"utt_id\tsim\n")
        for utt_id, sim in zip(manifest["id"][args.local_rank :: args.world_size], similarity_list):
            f.write(f"{utt_id}\t{sim}\n")

    print(f"Saved similarity results to {local_output_path}")


def load_merged_results(output_similarities):
    with open(output_similarities) as f:
        header = f.readline().strip()
        keys = header.split("\t")
        assert keys[0] == "utt_id"
        assert keys[1] == "sim"

        manifest = {k: [] for k in keys}
        for line in f:
            for k, v in zip(keys, line.strip().split("\t")):
                manifest[k].append(v)
    utt_id_list = manifest["utt_id"]
    similarity_list = [float(sim) for sim in manifest["sim"]]

    return utt_id_list, similarity_list


def aggregate_similarity_results(utt_list, similarity_list):
    overall_similarity = np.mean(similarity_list)

    speaker2similarity = defaultdict(list)
    accent2similarity = defaultdict(list)
    subset2similarity = {"seen": [], "unseen": []}

    for utt_id, sim in zip(utt_list, similarity_list):
        speaker = utt_id.split("_")[0]
        accent = SPEAKER_TO_ACCENT[speaker]
        subset = "unseen" if speaker in UNSEEN_SPEAKERS else "seen"

        speaker2similarity[speaker].append(sim)
        accent2similarity[accent].append(sim)
        subset2similarity[subset].append(sim)

    speaker2mean = {speaker: np.mean(sim_list) for speaker, sim_list in speaker2similarity.items()}
    accent2mean = {accent: np.mean(sim_list) for accent, sim_list in accent2similarity.items()}
    subset2mean = {subset: np.mean(sim_list) for subset, sim_list in subset2similarity.items()}

    return overall_similarity, speaker2mean, accent2mean, subset2mean


def get_parser():
    parser = argparse.ArgumentParser(description="Speech recognition evaluation script")
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to the metadata file",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="resemblyzer",
    )
    parser.add_argument(
        "--output_similarities",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_results",
        type=str,
        required=True,
    )
    parser.add_argument("--tag_to_recognize", type=str, default="gen_audio")
    parser.add_argument("--reference_tag", type=str, default="src_audio")
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

    os.makedirs(os.path.dirname(args.output_similarities), exist_ok=True)

    print(f"WORLD_SIZE: {args.world_size}, LOCAL_RANK: {args.local_rank}")
    print(f"#GPUs: {n_gpus}, GPU ID: {device_id}")

    with context:
        compute_similarity_list(args)

    if args.world_size > 1:
        torch.distributed.barrier()

    if args.local_rank == 0:
        output_manifest = merge_files(args.output_similarities, args.world_size)
        with open(args.output_similarities, "w") as f:
            f.writelines(output_manifest)

        utt_list, similarity_list = load_merged_results(args.output_similarities)
        overall_similarity, speaker2similarity, accent2similarity, subset2similarity = aggregate_similarity_results(
            utt_list, similarity_list
        )

        os.makedirs(os.path.dirname(args.output_results), exist_ok=True)
        with open(args.output_results, "w") as f:
            f.write(f"Overall SIM: {overall_similarity * 100:.2f}%\n")
            f.write("-----------Subset similarities-----------:\n")
            for sub, sim in subset2similarity.items():
                f.write(f"{sub}: {sim * 100:.2f}%\n")
            f.write("-----------Speaker similarities-----------:\n")
            for spk, sim in speaker2similarity.items():
                f.write(f"{spk}: {sim * 100:.2f}%\n")
            f.write("-----------Accent similarities-----------\n")
            for acc, sim in accent2similarity.items():
                f.write(f"{acc}: {sim * 100:.2f}%\n")

        print(f"Saved SIM results to {args.output_results}")
        print(f"Overall SIM: {overall_similarity * 100:.2f}%")
