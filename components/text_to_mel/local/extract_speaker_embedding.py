import os
import argparse
import pathlib
from tqdm import tqdm
from contextlib import nullcontext

from kaldiio import WriteHelper

import torch
import torchaudio

from resemblyzer import preprocess_wav, VoiceEncoder


def load_metadata(metadata_path, split="|"):
    metadata = []
    with open(metadata_path, "r") as f:
        for line in f:
            utt_id, wav_path, *res = line.strip().split(split)
            utt_id = os.path.splitext(os.path.basename(wav_path))[0]
            metadata.append((utt_id, wav_path, *res))
    return metadata


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def worker_shard_path(fname, worker_id):
    base_path = str(pathlib.Path(fname + f".partial_{worker_id}").absolute())
    scp_path = base_path + ".scp"
    arc_path = base_path + ".ark"
    return pathlib.Path(scp_path), pathlib.Path(arc_path)


def merge_scp(full_output, n_workers):
    with open(f"{full_output}.scp", "w") as full:
        for worker_id in tqdm(range(n_workers)):
            partial_path, _ = worker_shard_path(full_output, worker_id)
            partial_scp = open(partial_path, "r")
            for line in partial_scp:
                full.write(line)
            partial_path.unlink()


def extract(args):
    metadata = load_metadata(args.manifest, args.split)

    speaker_model = VoiceEncoder()

    shard_scp_path, shard_ark_path = worker_shard_path(args.output, args.local_rank)
    shard_ark_path.parent.mkdir(parents=True, exist_ok=True)
    ark_writer = WriteHelper(f"ark,scp:{shard_ark_path},{shard_scp_path}")

    for i in tqdm(range(args.local_rank, len(metadata), args.world_size)):
        utt_id, wav_path, *_ = metadata[i]

        wav = preprocess_wav(wav_path)
        embedding = speaker_model.embed_utterance(wav)
        embedding = torch.FloatTensor(embedding).unsqueeze(0).squeeze().cpu()

        ark_writer(utt_id, embedding.numpy())


def load_scp(scp_path):
    uid2path = {}
    with open(scp_path, "r") as f:
        for line in f:
            utt_id, emb_path = line.strip().split()
            uid2path[utt_id] = emb_path
    return uid2path


def write_metadata(input_manifest_path, spkemb_scp_path, output_manifest_path, split="|"):
    uid2path = load_scp(spkemb_scp_path)
    metadata = load_metadata(input_manifest_path, split)

    new_metadata = []
    for utt_id, wav_path, *res in metadata:
        if utt_id in uid2path:
            emb_path = uid2path[utt_id]
            new_metadata.append((utt_id, wav_path, *res, emb_path))

    with open(output_manifest_path, "w") as f:
        for entry in new_metadata:
            f.write(f"{split.join(entry)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to the original dataset manifest file")
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for the speaker embeddings, with suffix .scp and .ark"
    )
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to the output dataset manifest file")
    parser.add_argument("--split", type=str, default="|", help="Delimiter for manifest file")
    args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.distributed.init_process_group(backend="gloo", world_size=args.world_size, init_method="env://")
    n_gpus = torch.cuda.device_count()
    device_id = args.local_rank % n_gpus if n_gpus > 0 else None
    context = torch.cuda.device(device_id) if device_id is not None else nullcontext()

    print(f"WORLD_SIZE: {args.world_size}, LOCAL_RANK: {args.local_rank}")
    print(f"#GPUs: {n_gpus}, GPU ID: {device_id}")

    with context:
        extract(args)

    if args.world_size > 1:
        torch.distributed.barrier()

    if args.local_rank == 0:
        merge_scp(args.output, args.world_size)
        write_metadata(
            input_manifest_path=args.manifest,
            spkemb_scp_path=f"{args.output}.scp",
            output_manifest_path=args.output_manifest,
            split=args.split,
        )
