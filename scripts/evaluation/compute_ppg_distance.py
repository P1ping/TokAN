import os
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import torch
import ppgs

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
def compute_ppg_distance(tgt_audio_path, ref_audio_path, device_id=None):
    ref_audio = ppgs.load.audio(ref_audio_path)
    tgt_audio = ppgs.load.audio(tgt_audio_path)

    # Load PPGs (ensure they are NumPy arrays or convert if needed)
    ref_ppgs = ppgs.from_audio(ref_audio, ppgs.SAMPLE_RATE, gpu=device_id)[0].cpu().numpy().T  # Shape: (T1, dim)
    tgt_ppgs = ppgs.from_audio(tgt_audio, ppgs.SAMPLE_RATE, gpu=device_id)[0].cpu().numpy().T  # Shape: (T2, dim)

    # Align sequences using DTW
    distance, path = fastdtw(ref_ppgs, tgt_ppgs, dist=euclidean)

    # Warp sequences to align them (example: align tgt_ppgs to ref_ppgs length)
    aligned_ref = ref_ppgs[np.array([i for i, j in path])]
    aligned_tgt = tgt_ppgs[np.array([j for i, j in path])]

    # Convert aligned PPGs back to tensors
    aligned_ref_tensor = torch.tensor(aligned_ref).to(device_id)
    aligned_tgt_tensor = torch.tensor(aligned_tgt).to(device_id)

    # Compute distance
    ppg_dist = ppgs.distance(aligned_tgt_tensor.T, aligned_ref_tensor.T, reduction="mean", normalize=True)

    return ppg_dist.item()


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


def compute_ppg_distance_list(args):
    manifest = load_manifest(args.manifest_path, args.tag_to_recognize, args.reference_tag)

    distance_list = []
    for idx in tqdm(range(args.local_rank, len(manifest["id"]), args.world_size)):
        tgt_audio_path = manifest[args.tag_to_recognize][idx]
        ref_audio_path = manifest[args.reference_tag][idx]
        distance = compute_ppg_distance(tgt_audio_path, ref_audio_path, device_id=args.device_id)
        distance_list.append(distance)

    local_output_path = worker_shard_path(args.output_distances, args.local_rank)
    with open(local_output_path, "w") as f:
        f.write(f"utt_id\tppg_dist\n")
        for utt_id, dist in zip(manifest["id"][args.local_rank :: args.world_size], distance_list):
            f.write(f"{utt_id}\t{dist}\n")

    print(f"Saved PPG distance results to {local_output_path}")


def load_merged_results(output_distances):
    with open(output_distances) as f:
        header = f.readline().strip()
        keys = header.split("\t")
        assert keys[0] == "utt_id"
        assert keys[1] == "ppg_dist"

        manifest = {k: [] for k in keys}
        for line in f:
            for k, v in zip(keys, line.strip().split("\t")):
                manifest[k].append(v)
    utt_id_list = manifest["utt_id"]
    distance_list = [float(dist) for dist in manifest["ppg_dist"]]

    return utt_id_list, distance_list


def aggregate_distance_results(utt_list, distance_list):
    overall_distance = np.mean(distance_list)

    speaker2distance = defaultdict(list)
    accent2distance = defaultdict(list)
    subset2distance = {"seen": [], "unseen": []}

    for utt_id, dist in zip(utt_list, distance_list):
        speaker = utt_id.split("_")[0]
        accent = SPEAKER_TO_ACCENT[speaker]
        subset = "unseen" if speaker in UNSEEN_SPEAKERS else "seen"

        speaker2distance[speaker].append(dist)
        accent2distance[accent].append(dist)
        subset2distance[subset].append(dist)

    speaker2mean = {speaker: np.mean(dist_list) for speaker, dist_list in speaker2distance.items()}
    accent2mean = {accent: np.mean(dist_list) for accent, dist_list in accent2distance.items()}
    subset2mean = {subset: np.mean(dist_list) for subset, dist_list in subset2distance.items()}

    return overall_distance, speaker2mean, accent2mean, subset2mean


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to the metadata file",
    )
    parser.add_argument(
        "--output_distances",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_results",
        type=str,
        required=True,
    )
    parser.add_argument("--tag_to_recognize", type=str, default="gen_audio")
    parser.add_argument("--reference_tag", type=str, default="tgt_audio")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.distributed.init_process_group(backend="gloo", world_size=args.world_size, init_method="env://")
    os.makedirs(os.path.dirname(args.output_distances), exist_ok=True)

    n_gpus = torch.cuda.device_count()
    args.device_id = args.local_rank % n_gpus
    print(f"Local rank: {args.local_rank}, world size: {args.world_size}, n_gpus: {n_gpus}")
    compute_ppg_distance_list(args)

    if args.world_size > 1:
        torch.distributed.barrier()

    if args.local_rank == 0:
        output_manifest = merge_files(args.output_distances, args.world_size)
        with open(args.output_distances, "w") as f:
            f.writelines(output_manifest)

        utt_list, distance_list = load_merged_results(args.output_distances)
        overall_distance, speaker2distance, accent2distance, subset2distance = aggregate_distance_results(
            utt_list, distance_list
        )

        os.makedirs(os.path.dirname(args.output_results), exist_ok=True)
        with open(args.output_results, "w") as f:
            f.write(f"Overall PPG distance: {overall_distance}\n")
            f.write("-----------Subset PPG distances-----------:\n")
            for sub, dist in subset2distance.items():
                f.write(f"{sub}: {dist}\n")
            f.write("-----------Speaker PPG distances-----------:\n")
            for spk, dist in speaker2distance.items():
                f.write(f"{spk}: {dist}\n")
            f.write("-----------Accent PPG distances-----------\n")
            for acc, dist in accent2distance.items():
                f.write(f"{acc}: {dist}\n")

        print(f"Saved PPG distance results to {args.output_results}")
        print(f"Overall PPG distance: {overall_distance}")
