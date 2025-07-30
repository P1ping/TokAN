import os
import wave
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from contextlib import nullcontext

import soundfile as sf
from einops import rearrange

import torch
import torchaudio

from resemblyzer import VoiceEncoder, preprocess_wav

from tokan.utils.model_utils import load_bigvgan
from tokan.matcha.utils.model import denormalize
from tokan.matcha.utils.model import sequence_mask, fix_len_compatibility
from tokan.yirga.models.yirga_token_to_mel import YirgaTokenToMel


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        header = f.readline().strip()
        keys = header.split("\t")
        manifest = {k: [] for k in keys}
        for line in f:
            for k, v in zip(keys, line.strip().split("\t")):
                manifest[k].append(v)
    return manifest


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        if sample_rate < target_sr:
            print("Warning: wav sample rate {} is less than target sample rate {}".format(sample_rate, target_sr))
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def get_wav_duration(file_path):
    with wave.open(file_path, "r") as wav_file:
        n_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = n_frames / float(frame_rate)
    return duration


def load_model(ckpt, device):
    model = YirgaTokenToMel.load_from_checkpoint(ckpt, map_location=device)
    model.eval()
    return model


def deduplicate(tokens):
    dedup_tokens = []
    durations = []
    current_token = None
    current_count = 0

    for t in tokens:
        if t == current_token:
            current_count += 1
        else:
            if current_token is not None:
                dedup_tokens.append(current_token)
                durations.append(current_count)
            current_token = t
            current_count = 1

    # Append the last token and its count
    if current_token is not None:
        dedup_tokens.append(current_token)
        durations.append(current_count)

    return dedup_tokens, durations


@torch.inference_mode()
def synthesize_given_duration(
    model, x, x_lengths, duration, upsample_rate, spks, cond, n_timesteps, pitch=None, temperature=1.0, length_scale=1.0
):
    import torch.nn.functional as F
    from tokan.matcha.utils.model import sequence_mask, generate_path

    x, x_mask = model.encoder(x, x_lengths, spks)

    spk_embed = model.spk_embedder(spks)  # (B, D)
    x = x + spk_embed.unsqueeze(-1) * x_mask  # (B, D, T)

    durations = torch.ceil(duration * length_scale).long()  # (B, T)
    y_lengths = torch.sum(durations * upsample_rate, dim=1).long()
    y_max_length = y_lengths.max()

    x_dedup_lengths = durations.sum(dim=1).long()
    x_dedup_mask = sequence_mask(x_dedup_lengths, x_dedup_lengths.max()).unsqueeze(1).to(x_mask)
    attn_dedup_mask = x_mask.unsqueeze(-1) * x_dedup_mask.unsqueeze(2)

    attn_dedup = generate_path(durations.squeeze(1), attn_dedup_mask.squeeze(1))  # (B, T_x, T_y/k)
    attn = F.interpolate(attn_dedup.float(), scale_factor=upsample_rate, mode="nearest")  # (B, T_x, T_y)

    mel = synthesize_given_attn(model, x, y_lengths, attn, spks, cond, n_timesteps, temperature)

    output_batch = {
        "encoder_outputs": x,
        "attn": attn[:, :, :y_max_length],
        "mel": mel,
        "mel_lengths": y_lengths,
    }

    return output_batch


@torch.inference_mode()
def synthesize_given_attn(model, mu_x, y_lengths, attn, spks, cond, n_timesteps, temperature=1.0):
    y_max_length = y_lengths.max()
    y_max_length_ = fix_len_compatibility(y_max_length)
    y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).float()
    if attn.shape[-1] < y_max_length_:
        attn = torch.cat([attn, attn[:, :, -1].unsqueeze(-1).expand(-1, -1, y_max_length_ - attn.shape[-1])], dim=-1)

    # Align encoded text and get mu_y
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
    mu_y = mu_y.transpose(1, 2)

    # Generate sample tracing the probability flow
    decoder_outputs = model.decoder(mu_y, y_mask, n_timesteps, temperature, spks, cond=cond)
    decoder_outputs = decoder_outputs[:, :, :y_max_length]

    mel = denormalize(decoder_outputs, model.mel_mean, model.mel_std)

    return mel


@torch.inference_mode()
def extract_speaker_embedding(speaker_encoder, wav_path, device):
    speech = preprocess_wav(wav_path)
    spkemb = speaker_encoder.embed_utterance(speech)
    spkemb = torch.FloatTensor(spkemb).to(device)
    return spkemb


@torch.inference_mode()
def to_waveform(mel, vocoder):
    speech_synth = rearrange(vocoder(mel), "1 1 t -> 1 t").cpu().numpy()
    return speech_synth


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


@torch.inference_mode()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = load_manifest(args.manifest_path)

    speaker_encoder = VoiceEncoder(device=device)
    vocoder = load_bigvgan(args.bigvgan_tag_or_ckpt, device)
    sampling_rate = vocoder.h["sampling_rate"]
    hop_size = vocoder.h["hop_size"]

    model = load_model(args.yirga_ckpt, device)

    # Preserve original headers and add gen_audio column
    with open(args.manifest_path) as f:
        original_header = f.readline().strip()
    original_headers = original_header.split("\t")
    output_headers = original_headers + ["gen_audio"]
    output_manifest = ["\t".join(output_headers)]

    for i in tqdm(range(args.local_rank, len(manifest["id"]), args.world_size)):
        utt_id = manifest["id"][i]

        # Preserve all original columns
        original_row = []
        for header in original_headers:
            original_row.append(manifest[header][i])

        src_wav_path = manifest["src_audio"][i]

        src_tokens = manifest["src_tokens"][i]
        src_tokens = [int(t) for t in src_tokens.split()]
        src_tokens = torch.LongTensor(src_tokens).unsqueeze(0).to(device)

        gen_tokens = manifest["gen_tokens"][i]
        gen_tokens = [int(t) for t in gen_tokens.split()]
        gen_tokens, gen_durations = deduplicate(gen_tokens)
        gen_durations = torch.tensor(gen_durations, dtype=torch.long, device=device).unsqueeze(0)
        gen_tokens = torch.LongTensor(gen_tokens).unsqueeze(0).to(device)
        gen_lengths = torch.tensor([gen_tokens.size(1)], dtype=torch.long, device=device)

        spk_embed = extract_speaker_embedding(speaker_encoder, src_wav_path, device).unsqueeze(0)

        output_prefix = os.path.join(args.output_dir, f"{utt_id}")

        if args.source_duration_scale:
            total_wav_duration = get_wav_duration(src_wav_path) * args.source_duration_scale
            total_duration = torch.tensor([total_wav_duration * sampling_rate // hop_size]).long().to(device)
        else:
            total_duration = None

        if args.use_generated_duration:
            if torch.all(gen_durations == 1):
                print(
                    "Warning: All durations are 1 but `use_generated_duration` is set, maybe consider predicting durations?"
                )
            gen_output = synthesize_given_duration(
                model,
                gen_tokens,
                gen_lengths,
                gen_durations,
                upsample_rate=model.upsample_rate,
                spks=spk_embed,
                cond=None,
                n_timesteps=args.n_steps,
                temperature=args.temperature,
            )
        else:
            gen_output = model.synthesise(
                x=gen_tokens,
                x_lengths=gen_lengths,
                spks=spk_embed,
                cond=None,
                total_duration=total_duration,
                n_timesteps=args.n_steps,
                temperature=args.temperature,
                force_total_duration=args.force_total_duration,
            )

        gen_wav = to_waveform(gen_output["mel"], vocoder)
        output_path = output_prefix + ".gen.wav"
        sf.write(output_path, gen_wav.T, sampling_rate, "PCM_24")

        # Append the generated audio path to the original row
        output_row = original_row + [output_path]
        output_manifest.append("\t".join(output_row))

    with open(worker_shard_path(args.output_manifest_path, args.local_rank), "w") as f:
        for item in output_manifest:
            f.write(item + "\n")


def get_parser():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to the manifest")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output data directory")
    parser.add_argument("--output_manifest_path", type=str, required=True, help="Path to the output manifest")

    parser.add_argument("--yirga_ckpt", type=str, required=True, help="Checkpoint path of the Yirga token-to-mel model")
    parser.add_argument("--bigvgan_tag_or_ckpt", type=str, required=True, help="Vocoder checkpoint path")

    parser.add_argument("--temperature", type=float, default=0.95, help="Starting noise level for sampling")
    parser.add_argument("--n_steps", type=int, default=16, help="Steps for sampling")

    parser.add_argument(
        "--source_duration_scale", type=float, default=None, help="Scale factor for the source duration"
    )
    parser.add_argument(
        "--force_total_duration",
        action="store_true",
        help="Force the total duration to match the scaled source audio duration",
    )
    parser.add_argument(
        "--use_generated_duration",
        action="store_true",
        help="Use duration in the generated token sequences, indicating joint modeling of duration generation and token conversion",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.use_generated_duration and (args.source_duration_scale is not None):
        raise ValueError(
            "Cannot use both `use_generated_duration` and `source_duration_scale` at the same time. "
            "use_generated_duration indicates that duration is modeled by the token-to-token model, "
            "while source_duration_scale indicates the token-to-Mel model models the duration. "
            "Please choose one of them."
        )

    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.distributed.init_process_group(backend="gloo", world_size=args.world_size, init_method="env://")
    n_gpus = torch.cuda.device_count()
    device_id = args.local_rank % n_gpus if n_gpus > 0 else None
    context = torch.cuda.device(device_id) if device_id is not None else nullcontext()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_manifest_path), exist_ok=True)

    print(f"WORLD_SIZE: {args.world_size}, LOCAL_RANK: {args.local_rank}")
    print(f"#GPUs: {n_gpus}, GPU ID: {device_id}")

    with context:
        main(args)

    if args.world_size > 1:
        torch.distributed.barrier()

    if args.local_rank == 0:
        output_manifest = merge_files(args.output_manifest_path, args.world_size)
        with open(args.output_manifest_path, "w") as f:
            f.writelines(output_manifest)
