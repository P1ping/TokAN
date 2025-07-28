import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from einops import rearrange
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
import torch.nn.functional as F
from torch import Tensor

from tokan.utils.model_utils import ModelManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio processing, segmentation and VAD."""

    def __init__(self, device: torch.device):
        self.device = device
        self._vad_model = None

    @property
    def vad_model(self):
        """Lazy-load the VAD model."""
        if self._vad_model is None:
            from silero_vad import load_silero_vad

            self._vad_model = load_silero_vad()
        return self._vad_model

    def load_audio(self, path: str, target_sr: int = 16000) -> Tensor:
        """Load audio file and resample if necessary."""
        from silero_vad import read_audio

        return read_audio(path, sampling_rate=target_sr)

    def segment_audio(
        self, wav: Tensor, threshold: float = 0.5, max_duration: float = 10.0, min_duration: float = 3.0
    ) -> List[Dict[str, float]]:
        """Segment audio using VAD and reorganize chunks."""
        from silero_vad import get_speech_timestamps

        # Get initial speech timestamps
        speech_timestamps = get_speech_timestamps(wav, self.vad_model, return_seconds=True, threshold=threshold)

        # Reorganize to meet duration constraints
        return self._reorganize_chunks(speech_timestamps, max_duration, min_duration)

    def _reorganize_chunks(
        self, speech_timestamps: List[Dict[str, float]], max_duration: float = 10.0, min_duration: float = 3.0
    ) -> List[Dict[str, float]]:
        """Reorganize speech chunks to meet duration constraints."""
        if not speech_timestamps:
            return []

        reorganized_chunks = []
        i = 0

        while i < len(speech_timestamps):
            chunk = speech_timestamps[i]
            duration = chunk["end"] - chunk["start"]

            # Case 1: Chunk is too long
            if duration > max_duration:
                logger.warning(
                    f"Chunk {i} exceeds max_duration ({duration:.2f}s). "
                    f"Consider re-running VAD with higher threshold."
                )
                reorganized_chunks.append(chunk)
                i += 1
                continue

            # Case 2: Chunk is too short - try to combine with the next chunk
            if duration < min_duration and i < len(speech_timestamps) - 1:
                next_chunk = speech_timestamps[i + 1]
                combined_duration = next_chunk["end"] - chunk["start"]

                # Check if combining would still keep us under max_duration
                if combined_duration <= max_duration:
                    reorganized_chunks.append({"start": chunk["start"], "end": next_chunk["end"]})
                    i += 2  # Skip the next chunk since we've combined it
                else:
                    # If combining would make it too long, keep the short chunk
                    reorganized_chunks.append(chunk)
                    i += 1

            # Default case: Chunk is within acceptable limits or it's the last chunk
            else:
                reorganized_chunks.append(chunk)
                i += 1

        return reorganized_chunks

    def split_waveform(
        self, wav: Tensor, timestamps: List[Dict[str, float]], sr: int = 16000, include_silence: bool = False
    ) -> List[Tensor]:
        """Split waveform according to timestamps."""
        if len(timestamps) <= 1:
            return [wav]

        if include_silence:
            start_seconds = [0.0] + [chunk["end"] for chunk in timestamps[:-1]]
            end_seconds = [chunk["start"] for chunk in timestamps[1:]] + [wav.shape[-1] / sr]
        else:
            start_seconds = [chunk["start"] for chunk in timestamps]
            end_seconds = [chunk["end"] for chunk in timestamps]

        wav_list = []
        for idx, chunk in enumerate(timestamps):
            start = int(start_seconds[idx] * sr)
            end = int(end_seconds[idx] * sr)
            wav_seg = wav[start:end]
            wav_list.append(wav_seg)

        return wav_list

    def concatenate_waveforms(
        self, wav_list: List[Tensor], source_timestamps: List[Dict[str, float]], sampling_rate: int
    ) -> Tensor:
        """Concatenate waveforms with silence based on timestamps."""
        assert len(wav_list) == len(source_timestamps), "Mismatch between wav_list and timestamps length"

        concat_list = []
        prev_end = 0.0

        for wav, chunk in zip(wav_list, source_timestamps):
            sil_duration = chunk["start"] - prev_end
            if sil_duration > 0:
                silence = torch.zeros(int(sil_duration * sampling_rate)).to(self.device)
                concat_list.append(silence)
            concat_list.append(wav.squeeze().to(self.device))
            prev_end = chunk["end"]

        wav = torch.cat(concat_list, dim=-1)
        return wav.unsqueeze(0)  # (1, T)


class FeatureExtractor:
    """Extracts speaker, accent and content features from audio."""

    def __init__(self, device: torch.device):
        self.device = device
        self._speaker_encoder = None
        self._accent_model = None
        self._hubert = None
        self._kmeans = None
        self._speech_encoder = None

    def extract_speaker_embedding(self, wav: np.ndarray) -> Tensor:
        """Extract speaker embedding from waveform."""
        if self._speaker_encoder is None:
            from resemblyzer import VoiceEncoder

            self._speaker_encoder = VoiceEncoder()

        spk_embed = self._speaker_encoder.embed_utterance(wav)
        return torch.FloatTensor(spk_embed).unsqueeze(0).to(self.device)  # (1, D)

    def extract_accent_embedding(self, wav: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract accent embedding and probabilities from waveform."""
        if self._accent_model is None:
            from speechbrain.pretrained.interfaces import foreign_class

            self._accent_model = foreign_class(
                source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=None,
                run_opts={"device": self.device},
            )

        batch = wav.unsqueeze(0).to(self._accent_model.device)
        rel_length = torch.tensor([1.0])
        embeds = self._accent_model.encode_batch(batch, rel_length)
        embeds = embeds.squeeze(dim=1)  # (B, D)

        proj = (
            self._accent_model.mods.classifier
            if hasattr(self._accent_model.mods, "classifier")
            else self._accent_model.mods.output_mlp
        )
        logits = proj(embeds)  # (B, N_cls)
        probs = F.softmax(logits, dim=-1)  # (B, N_cls)

        return F.layer_norm(embeds, embeds.shape), probs

    def setup_content_encoder(self, hubert_path: str, hubert_layer: int, kmeans_path: str) -> None:
        """Set up the content encoder with HuBERT and K-means models."""
        from tokan.textless.hubert_feature_reader import HubertFeatureReader
        from tokan.textless.kmeans_quantizer import KMeansQuantizer
        from tokan.textless.speech_encoder import SpeechEncoder

        self._hubert = HubertFeatureReader(hubert_path, layer=hubert_layer)
        self._kmeans = KMeansQuantizer(kmeans_path)
        self._speech_encoder = SpeechEncoder(
            self._hubert,
            self._kmeans,
            need_f0=False,
            deduplicate=True,
            padding=True,
        ).cuda()

    def extract_content_tokens(self, wav_list: List[Tensor]) -> List[Tensor]:
        """Extract content tokens from a list of waveforms."""
        if self._speech_encoder is None:
            raise RuntimeError("Content encoder not initialized. Call setup_content_encoder first.")

        token_list = []
        for wav in tqdm(wav_list, desc="Extracting content tokens"):
            encoder_out = self._speech_encoder(wav)
            tokens = encoder_out["units"].squeeze(0)
            token_list.append(tokens)

        return token_list


class TokenConverter:
    """Converts source tokens to target tokens."""

    def __init__(self, conversion_model_path: str, device: torch.device):
        self.device = device
        self.conversion_model_path = conversion_model_path
        self._models = None
        self._task = None
        self._generator = None
        self._src_dict = None
        self._tgt_dict = None

    def setup(self, beam_size: int = 10, seed: int = 1337) -> None:
        """Set up the conversion model and generator."""
        from fairseq import checkpoint_utils, utils as fairseq_utils

        # Check if the fairseq_modules module can be imported
        try:
            fairseq_utils.import_user_module(argparse.Namespace(user_dir="tokan/fairseq_modules"))
        except ImportError:
            # If not, create a dummy module for backward compatibility
            logger.warning("fairseq_modules module not found, using fallback approach")
            import sys
            from types import ModuleType

            module = ModuleType("fairseq_modules")
            sys.modules["fairseq_modules"] = module

        # Load models and task
        self._models, task_cfg, self._task = checkpoint_utils.load_model_ensemble_and_task(
            [self.conversion_model_path], task=None
        )

        self._models[0].to(self.device).eval()
        self._src_dict = self._task.source_dictionary
        self._tgt_dict = self._task.target_dictionary

        # Configure generator
        task_cfg.generation.beam = beam_size
        np.random.seed(seed)
        fairseq_utils.set_torch_seed(seed)

        self._generator = self._task.build_generator(self._models, task_cfg.generation)

    def convert_tokens(self, token_list: List[Tensor], accent_embedding: Tensor) -> List[Tensor]:
        """Convert source tokens to target tokens using the conversion model."""
        if self._generator is None:
            raise RuntimeError("Converter not initialized. Call setup first.")

        converted_token_list = []

        for tokens in tqdm(token_list, desc="Converting tokens"):
            # Prepare input
            src_token_str = " ".join([str(t) for t in tokens.tolist()])
            input_tokens = self._src_dict.encode_line(src_token_str, add_if_not_exist=False, append_eos=False).long()
            src_tokens = input_tokens.unsqueeze(0).to(self.device)  # (1, T)
            src_lengths = torch.tensor([src_tokens.shape[1]]).to(self.device)  # (1)

            # Run model
            net_input = {"src_tokens": src_tokens, "src_lengths": src_lengths, "condition_embeds": accent_embedding}
            sample = {"net_input": net_input}

            hypos = self._task.inference_step(
                self._generator, self._models, sample, prefix_tokens=None, constraints=None
            )

            # Process output
            hypo = hypos[0][0]  # [0] is the first hypothesis, [0] is the first beam
            hypo_str = self._tgt_dict.string(hypo["tokens"].int().cpu())
            converted_token_list.append(torch.LongTensor([int(t) for t in hypo_str.split()]).to(self.device))

        return converted_token_list


class MelSynthesizer:
    """Synthesizes mel spectrograms from tokens."""

    def __init__(self, synthesis_model_path: str, device: torch.device):
        self.synthesis_model_path = synthesis_model_path
        self.device = device
        self._model = None

    def setup(self) -> None:
        """Load the synthesis model."""
        from tokan.yirga.models.yirga_token_to_mel import YirgaTokenToMel

        self._model = YirgaTokenToMel.load_from_checkpoint(self.synthesis_model_path, map_location=self.device)
        self._model.to(self.device).eval()

    def synthesize(
        self,
        token_list: List[Tensor],
        speaker_embedding: Tensor,
        duration_list: Optional[List[float]] = None,
        preserve_duration: bool = False,
        hop_size: int = 256,
        sampling_rate: int = 22050,
    ) -> List[Tensor]:
        """Synthesize mel spectrograms from tokens."""
        if self._model is None:
            raise RuntimeError("Synthesizer not initialized. Call setup first.")

        mel_list = []

        for idx, tokens in tqdm(enumerate(token_list), desc="Synthesizing mel spectrograms", total=len(token_list)):
            # Calculate duration if needed
            total_duration = None
            if preserve_duration and duration_list is not None:
                total_duration = torch.tensor(duration_list[idx] * sampling_rate // hop_size).long().to(self.device)

            # Run model
            t2m_output = self._model.synthesise(
                x=tokens.unsqueeze(0),
                x_lengths=torch.tensor([tokens.shape[0]]).long().to(self.device),
                spks=speaker_embedding,
                cond=None,
                total_duration=total_duration,
                n_timesteps=32,
                temperature=0.99,
            )

            mel = t2m_output["mel"]  # (1, D_Mel, T_Mel)
            mel_list.append(mel)

        return mel_list


class Vocoder:
    """Converts mel spectrograms to waveforms."""

    def __init__(self, bigvgan_tag_or_ckpt: str, device: torch.device):
        self.bigvgan_tag_or_ckpt = bigvgan_tag_or_ckpt
        self.device = device
        self._vocoder = None

    def hop_size(self) -> int:
        """Return the hop size of the vocoder."""
        return self._vocoder.h["hop_size"]

    def sampling_rate(self) -> int:
        """Return the sampling rate of the vocoder."""
        return self._vocoder.h["sampling_rate"]

    def setup(self) -> None:
        """Load the vocoder model."""
        from tokan.utils.model_utils import load_bigvgan

        self._vocoder = load_bigvgan(self.bigvgan_tag_or_ckpt, device=self.device)

    def generate_waveforms(self, mel_list: List[Tensor]) -> List[Tensor]:
        """Generate waveforms from mel spectrograms."""
        if self._vocoder is None:
            raise RuntimeError("Vocoder not initialized. Call setup first.")

        wav_list = []
        for mel in tqdm(mel_list, desc="Generating waveforms"):
            wav = rearrange(self._vocoder(mel), "1 1 t -> 1 t")
            wav_list.append(wav)

        return wav_list


class AccentConverter:
    """Main class for accent conversion."""

    def __init__(self, config: Dict):
        """Initialize accent converter with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.audio_processor = AudioProcessor(self.device)
        self.feature_extractor = FeatureExtractor(self.device)
        self.token_converter = TokenConverter(config["conversion_model"], self.device)
        self.mel_synthesizer = MelSynthesizer(config["synthesis_model"], self.device)
        self.vocoder = Vocoder(config["bigvgan_tag_or_ckpt"], self.device)

    def setup(self) -> None:
        """Set up all components."""
        logger.info("Setting up accent converter components...")

        # Set up content encoder
        self.feature_extractor.setup_content_encoder(
            self.config["hubert_model"], self.config["hubert_layer"], self.config["hubert_km_path"]
        )

        # Set up token converter
        self.token_converter.setup(beam_size=self.config["beam_size"])

        # Set up synthesizer and vocoder
        self.mel_synthesizer.setup()
        self.vocoder.setup()

        logger.info("Setup complete!")

    def convert(self, input_path: str, output_path: str) -> None:
        """Convert accent in input file and save to output file."""
        logger.info(f"Converting accent in {input_path}")

        # Step 1: Load and segment audio
        wav = self.audio_processor.load_audio(input_path)
        timestamps = self.audio_processor.segment_audio(
            wav,
            threshold=self.config["vad_threshold"],
            max_duration=self.config["max_duration"],
            min_duration=self.config["min_duration"],
        )

        if not timestamps:
            logger.warning("No speech detected in the input file!")
            return

        logger.info(f"Segmented audio into {len(timestamps)} chunks")
        wav_list = self.audio_processor.split_waveform(wav, timestamps)

        # Step 2: Extract embedding from longest segments
        duration_list = [chunk["end"] - chunk["start"] for chunk in timestamps]
        duration_sorted_indices = np.argsort(duration_list)[::-1]

        wavs_for_embedding = []
        duration_for_embedding = 0.0

        for i in duration_sorted_indices:
            if duration_for_embedding > self.config["min_embed_duration"]:
                break
            wavs_for_embedding.append(wav_list[i])
            duration_for_embedding += duration_list[i]

        wavs_for_embedding = torch.concat(wavs_for_embedding, dim=0)

        # Step 3: Extract features
        logger.info("Extracting speaker and accent features")
        spk_embed = self.feature_extractor.extract_speaker_embedding(wavs_for_embedding.numpy())
        act_embed, _ = self.feature_extractor.extract_accent_embedding(wavs_for_embedding)

        logger.info("Extracting content tokens")
        token_list = self.feature_extractor.extract_content_tokens(wav_list)

        # Step 4: Convert tokens
        converted_token_list = self.token_converter.convert_tokens(token_list, act_embed)

        # Step 5: Synthesize mel spectrograms
        mel_list = self.mel_synthesizer.synthesize(
            converted_token_list,
            spk_embed,
            duration_list=duration_list,
            preserve_duration=self.config["preserve_duration"],
            hop_size=self.vocoder.hop_size(),
            sampling_rate=self.vocoder.sampling_rate(),
        )

        # Step 6: Generate waveforms and concatenate
        converted_wav_list = self.vocoder.generate_waveforms(mel_list)
        converted_wav = self.audio_processor.concatenate_waveforms(
            converted_wav_list, timestamps, self.vocoder.sampling_rate()
        )

        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, converted_wav.cpu(), self.vocoder.sampling_rate())
        logger.info(f"Converted audio saved to {output_path}")

        # Save debug outputs if verbose mode is enabled
        if self.config.get("verbose", False):
            self._save_debug_outputs(
                wav_list, converted_wav_list, token_list, converted_token_list, timestamps, output_path
            )

    def _save_debug_outputs(
        self,
        wav_list: List[Tensor],
        converted_wav_list: List[Tensor],
        token_list: List[Tensor],
        converted_token_list: List[Tensor],
        timestamps: List[Dict[str, float]],
        output_path: str,
    ) -> None:
        """Save debug outputs for analysis."""
        save_dir = os.path.dirname(output_path)
        debug_dir = os.path.join(save_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        tok_file = open(os.path.join(debug_dir, "tokens.txt"), "w")

        for idx, (original_wav, converted_wav) in enumerate(zip(wav_list, converted_wav_list)):
            # Save audio segments
            original_wav = original_wav.unsqueeze(0).cpu()
            converted_wav = converted_wav.cpu()
            torchaudio.save(os.path.join(debug_dir, f"original_{idx}.wav"), original_wav, 16000)
            torchaudio.save(
                os.path.join(debug_dir, f"converted_{idx}.wav"), converted_wav, self.vocoder.sampling_rate()
            )

            # Save token information
            orig_token_str = " ".join([str(t) for t in token_list[idx].tolist()])
            conv_token_str = " ".join([str(t) for t in converted_token_list[idx].tolist()])
            tok_file.write(
                f"{idx} | {timestamps[idx]['start']:.1f}-{timestamps[idx]['end']:.1f} | {orig_token_str} | {conv_token_str}\n"
            )

        tok_file.close()
        logger.info(f"Debug outputs saved to {debug_dir}")


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Accent Conversion Tool")

    # Input/output arguments
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--output_path", type=str, default="./output.wav", help="Path to the converted audio file")

    # Model paths (will be downloaded if not specified)
    parser.add_argument("--hubert_model", type=str, help="Path to the HuBERT model")
    parser.add_argument("--hubert_km_path", type=str, help="Path to the HuBERT K-Means model")
    parser.add_argument("--conversion_model", type=str, help="Path to the conversion model")
    parser.add_argument("--synthesis_model", type=str, help="Path to the synthesis model")
    parser.add_argument("--bigvgan_tag_or_ckpt", type=str, help="Path or Huggingface tag to the BigVGAN checkpoint")

    parser.add_argument("--preserve_duration", action="store_true", help="Preserve total duration during conversion")

    # Feature extraction settings
    parser.add_argument("--hubert_layer", type=int, default=17, help="Layer to extract features from")
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size for decoding")

    # Segmentation settings
    parser.add_argument("--vad_threshold", type=float, default=0.5, help="VAD threshold for chunking")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Max duration for a chunk")
    parser.add_argument("--min_duration", type=float, default=3.0, help="Min duration for a chunk")
    parser.add_argument("--min_embed_duration", type=float, default=5.0, help="Min duration for embedding extraction")

    # Other settings
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output and debug files")
    parser.add_argument("--download_models", action="store_true", help="Download models if not found locally")

    return parser


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args()

    # Check if models should be downloaded
    if args.download_models:
        logger.info("Checking for model files...")

        # Get model paths, downloading if necessary
        hubert_model = args.hubert_model or ModelManager.ensure_model_available("hubert")
        hubert_km_path = args.hubert_km_path or ModelManager.ensure_model_available("hubert_km")
        conversion_model = args.conversion_model or ModelManager.ensure_model_available("token_to_token")
        # By default, the flow-matching-based duration predictor (v2) is used for total duration preservation
        # In fact, both v1 and v2 can be used
        synthesis_model = args.synthesis_model or ModelManager.ensure_model_available(
            "token_to_mel_v2" if args.preserve_duration else "token_to_mel_v1"
        )
        bigvgan_model = args.bigvgan_tag_or_ckpt or "nvidia/bigvgan_22khz_80band"
    else:
        # Use specified paths
        hubert_model = args.hubert_model
        hubert_km_path = args.hubert_km_path
        conversion_model = args.conversion_model
        synthesis_model = args.synthesis_model
        bigvgan_model = args.bigvgan_tag_or_ckpt

        # Check if all required models are specified
        if not all([hubert_model, hubert_km_path, conversion_model, synthesis_model, bigvgan_model]):
            parser.error("All model paths must be specified or use --download_models")

    # Create configuration
    config = {
        "hubert_model": hubert_model,
        "hubert_layer": args.hubert_layer,
        "hubert_km_path": hubert_km_path,
        "conversion_model": conversion_model,
        "beam_size": args.beam_size,
        "synthesis_model": synthesis_model,
        "bigvgan_tag_or_ckpt": bigvgan_model,
        "preserve_duration": args.preserve_duration,
        "vad_threshold": args.vad_threshold,
        "max_duration": args.max_duration,
        "min_duration": args.min_duration,
        "min_embed_duration": args.min_embed_duration,
        "verbose": args.verbose,
    }

    # Create and set up accent converter
    converter = AccentConverter(config)
    converter.setup()

    # Perform conversion
    with torch.inference_mode():
        converter.convert(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
