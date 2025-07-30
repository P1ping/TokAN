#!/usr/bin/env python3
"""
Data Preparation Verification Script for TokAN
Verifies that LibriTTS-R and L2ARCTIC datasets are properly prepared for training.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import logging

from tokan.data.l2arctic import ACCENT_TO_SPEAKER

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# LibriTTS-R expected subsets
LIBRITTS_SUBSETS = [
    "train-other-500",
    "train-clean-360",
    "train-clean-100",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]


class DataVerifier:
    def __init__(self, libritts_root, l2arctic_root):
        self.libritts_root = Path(libritts_root)
        self.l2arctic_root = Path(l2arctic_root)
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

    def log_error(self, message):
        """Log an error message"""
        self.errors.append(message)
        logger.error(f"❌ {message}")

    def log_warning(self, message):
        """Log a warning message"""
        self.warnings.append(message)
        logger.warning(f"⚠️  {message}")

    def log_success(self, message):
        """Log a success message"""
        logger.info(f"✅ {message}")

    def log_info(self, message):
        """Log an info message"""
        logger.info(f"ℹ️  {message}")

    def verify_libritts(self):
        """Verify LibriTTS-R dataset structure and completeness"""
        logger.info("=" * 60)
        logger.info("VERIFYING LIBRITTS-R DATASET")
        logger.info("=" * 60)

        if not self.libritts_root.exists():
            self.log_error(f"LibriTTS-R root directory not found: {self.libritts_root}")
            return False

        self.log_info(f"LibriTTS-R root: {self.libritts_root}")

        missing_subsets = []
        total_audio_files = 0
        total_text_files = 0

        for subset in LIBRITTS_SUBSETS:
            subset_dir = self.libritts_root / subset

            if not subset_dir.exists():
                missing_subsets.append(subset)
                self.log_error(f"Missing subset: {subset}")
                continue

            # Count audio and text files
            audio_files = list(subset_dir.glob("**/*.wav"))
            text_files = list(subset_dir.glob("**/*.normalized.txt"))

            # Count speakers (subdirectories with numeric names)
            speakers = [d for d in subset_dir.iterdir() if d.is_dir() and d.name.isdigit()]

            self.log_success(
                f"{subset}: {len(audio_files)} audio files, {len(text_files)} text files, {len(speakers)} speakers"
            )

            total_audio_files += len(audio_files)
            total_text_files += len(text_files)
            self.stats[f"libritts_{subset}_audio"] = len(audio_files)
            self.stats[f"libritts_{subset}_text"] = len(text_files)
            self.stats[f"libritts_{subset}_speakers"] = len(speakers)

            # Verify some basic structure
            if len(audio_files) == 0:
                self.log_warning(f"No audio files found in {subset}")
            if len(text_files) == 0:
                self.log_warning(f"No text files found in {subset}")
            if len(audio_files) != len(text_files):
                self.log_warning(
                    f"Mismatch between audio ({len(audio_files)}) and text ({len(text_files)}) files in {subset}"
                )

        if missing_subsets:
            self.log_error(f"Missing LibriTTS-R subsets: {missing_subsets}")
            logger.info("\n" + "📋 HOW TO DOWNLOAD LIBRITTS-R")
            logger.info("1. Visit: https://www.openslr.org/141/")
            logger.info("2. Download the required subsets:")
            for subset in missing_subsets:
                logger.info(f"   - {subset}.tar.gz")
            logger.info("3. Extract each subset to your LibriTTS-R directory")
            logger.info(f"4. Expected structure: {self.libritts_root}/[subset_name]/")
            return False
        else:
            self.log_success(
                f"All LibriTTS-R subsets found! Total: {total_audio_files} audio files, {total_text_files} text files"
            )
            self.stats["libritts_total_audio"] = total_audio_files
            self.stats["libritts_total_text"] = total_text_files
            return True

    def verify_l2arctic_speakers(self):
        """Verify L2ARCTIC and ARCTIC speakers"""
        logger.info("=" * 60)
        logger.info("VERIFYING L2ARCTIC AND ARCTIC SPEAKERS")
        logger.info("=" * 60)

        if not self.l2arctic_root.exists():
            self.log_error(f"L2ARCTIC root directory not found: {self.l2arctic_root}")
            return False

        self.log_info(f"L2ARCTIC root: {self.l2arctic_root}")

        all_speakers = []
        for accent, speakers in ACCENT_TO_SPEAKER.items():
            all_speakers.extend(speakers)

        missing_speakers = []
        found_speakers = []
        speaker_stats = {}

        for accent, speakers in ACCENT_TO_SPEAKER.items():
            self.log_info(f"\nChecking {accent} speakers: {speakers}")

            for speaker in speakers:
                # Check multiple possible locations for the speaker
                possible_paths = [
                    self.l2arctic_root / speaker,
                ]

                speaker_found = False
                speaker_path = None

                for path in possible_paths:
                    if path.exists() and path.is_dir():
                        speaker_found = True
                        speaker_path = path
                        break

                if speaker_found:
                    # Count audio files
                    audio_files = list(speaker_path.glob("**/*.wav"))
                    found_speakers.append(speaker)
                    speaker_stats[speaker] = {"path": speaker_path, "audio_files": len(audio_files), "accent": accent}
                    self.log_success(f"  {speaker}: {len(audio_files)} audio files at {speaker_path}")
                else:
                    missing_speakers.append(speaker)
                    self.log_error(f"  {speaker}: NOT FOUND")

        # Summary by accent
        logger.info("\n" + "=" * 40)
        logger.info("SPEAKER SUMMARY BY ACCENT")
        logger.info("=" * 40)

        for accent, speakers in ACCENT_TO_SPEAKER.items():
            found_count = sum(1 for s in speakers if s in found_speakers)
            total_count = len(speakers)
            accent_audio_count = sum(
                speaker_stats.get(s, {}).get("audio_files", 0) for s in speakers if s in found_speakers
            )

            if found_count == total_count:
                self.log_success(
                    f"{accent}: {found_count}/{total_count} speakers, {accent_audio_count} total audio files"
                )
            else:
                self.log_error(
                    f"{accent}: {found_count}/{total_count} speakers, {accent_audio_count} total audio files"
                )

            self.stats[f"speakers_{accent}_found"] = found_count
            self.stats[f"speakers_{accent}_total"] = total_count
            self.stats[f"speakers_{accent}_audio"] = accent_audio_count

        if missing_speakers:
            self.log_error(f"\nMissing speakers: {missing_speakers}")

            # Provide specific guidance
            l2arctic_missing = [s for s in missing_speakers if s not in ACCENT_TO_SPEAKER["<us>"]]
            arctic_missing = [s for s in missing_speakers if s in ACCENT_TO_SPEAKER["<us>"]]

            if l2arctic_missing:
                logger.info("\n" + "📋 MISSING L2ARCTIC SPEAKERS")
                logger.info("To obtain L2ARCTIC corpus:")
                logger.info("1. Visit: https://psi.engr.tamu.edu/l2-arctic-corpus/")
                logger.info("2. Submit a request form with your research details")
                logger.info("3. You will receive a Google Drive link after approval")
                logger.info("4. Download and extract to your L2ARCTIC root directory")
                logger.info("5. Ensure the following speakers are present:")
                for accent, speakers in ACCENT_TO_SPEAKER.items():
                    if accent != "<us>":  # Skip US speakers as they're from ARCTIC
                        missing_in_accent = [s for s in speakers if s in l2arctic_missing]
                        if missing_in_accent:
                            logger.info(f"   {accent}: {missing_in_accent}")
                logger.info(f"6. Expected structure: {self.l2arctic_root}/[SPEAKER_ID]/")

            if arctic_missing:
                logger.info("\n" + "📋 MISSING ARCTIC NATIVE SPEAKERS")
                logger.info("To download ARCTIC native speakers:")
                logger.info("1. Visit: http://festvox.org/cmu_arctic/")
                logger.info("2. Download the following speakers:")
                for speaker in arctic_missing:
                    speaker_lower = speaker.lower()
                    logger.info(f"   - cmu_us_{speaker_lower}_arctic.tar.bz2")
                logger.info("3. Extract to your L2ARCTIC directory")
                logger.info(f"4. Expected paths: {self.l2arctic_root}/[SPEAKER]/ in the upper case")

            return False
        else:
            self.log_success(f"All speakers found! Total: {len(found_speakers)} speakers")
            total_l2arctic_audio = sum(
                self.stats.get(f"speakers_{accent}_audio", 0) for accent in ACCENT_TO_SPEAKER.keys()
            )
            self.stats["l2arctic_total_audio"] = total_l2arctic_audio
            return True

    def verify_additional_requirements(self):
        """Verify additional requirements like HuBERT models"""
        logger.info("=" * 60)
        logger.info("VERIFYING ADDITIONAL REQUIREMENTS")
        logger.info("=" * 60)

        # Check for HuBERT model paths from run.sh
        hubert_path = "/dockerdata/qibingbai/.textless/hubert_large_ll60k.pt"
        km_path = "/dockerdata/qibingbai/.textless/hubert-large_l17_km-librittsr-1000.pt"

        if Path(hubert_path).exists():
            self.log_success(f"HuBERT model found: {hubert_path}")
        else:
            self.log_warning(f"HuBERT model not found: {hubert_path}")
            logger.info("  This model will be downloaded automatically during training")

        if Path(km_path).exists():
            self.log_success(f"K-means model found: {km_path}")
        else:
            self.log_warning(f"K-means model not found: {km_path}")
            logger.info("  This model will be generated during k-means clustering stage")

    def print_statistics(self):
        """Print detailed statistics"""
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)

        # LibriTTS-R statistics
        if self.stats.get("libritts_total_audio", 0) > 0:
            logger.info(f"📊 LibriTTS-R Total: {self.stats['libritts_total_audio']} audio files")
            for subset in LIBRITTS_SUBSETS:
                audio_count = self.stats.get(f"libritts_{subset}_audio", 0)
                speaker_count = self.stats.get(f"libritts_{subset}_speakers", 0)
                if audio_count > 0:
                    logger.info(f"   {subset}: {audio_count} files, {speaker_count} speakers")

        # L2ARCTIC statistics
        if self.stats.get("l2arctic_total_audio", 0) > 0:
            logger.info(f"📊 L2ARCTIC/ARCTIC Total: {self.stats['l2arctic_total_audio']} audio files")
            for accent in ACCENT_TO_SPEAKER.keys():
                found = self.stats.get(f"speakers_{accent}_found", 0)
                total = self.stats.get(f"speakers_{accent}_total", 0)
                audio = self.stats.get(f"speakers_{accent}_audio", 0)
                if total > 0:
                    logger.info(f"   {accent}: {found}/{total} speakers, {audio} files")

        # Calculate total dataset size
        total_audio = self.stats.get("libritts_total_audio", 0) + self.stats.get("l2arctic_total_audio", 0)
        if total_audio > 0:
            logger.info(f"🎵 Grand Total: {total_audio} audio files across all datasets")

    def run_verification(self):
        """Run complete verification"""
        logger.info("🔍 Starting TokAN Data Preparation Verification")
        logger.info(f"Timestamp: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}")

        # Verify LibriTTS-R
        libritts_ok = self.verify_libritts()

        # Verify L2ARCTIC speakers
        l2arctic_ok = self.verify_l2arctic_speakers()

        # Check additional requirements
        self.verify_additional_requirements()

        # Print statistics
        self.print_statistics()

        # Final summary
        logger.info("=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)

        if libritts_ok and l2arctic_ok:
            self.log_success("All datasets are properly prepared! ✨")
            self.log_info("You can proceed with training using: bash run.sh --stage 0")
            return True
        else:
            if self.errors:
                logger.error("❌ ERRORS FOUND:")
                for error in self.errors:
                    logger.error(f"   • {error}")

            if self.warnings:
                logger.warning("⚠️  WARNINGS:")
                for warning in self.warnings:
                    logger.warning(f"   • {warning}")

            self.log_error("Data preparation incomplete. Please address the issues above.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify TokAN dataset preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_data_preparation.py
  python scripts/verify_data_preparation.py --libritts_root /custom/path/LibriTTS_R
  python scripts/verify_data_preparation.py --l2arctic_root /custom/path/l2arctic --verbose
        """,
    )

    parser.add_argument(
        "--libritts_root",
        type=str,
        default="/dockerdata/qibingbai/datasets/LibriTTS_R",
        help="Path to LibriTTS-R root directory (default: /dockerdata/qibingbai/datasets/LibriTTS_R)",
    )

    parser.add_argument(
        "--l2arctic_root",
        type=str,
        default="/dockerdata/qibingbai/datasets/l2arctic",
        help="Path to L2ARCTIC root directory (default: /dockerdata/qibingbai/datasets/l2arctic)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create verifier and run
    verifier = DataVerifier(args.libritts_root, args.l2arctic_root)
    success = verifier.run_verification()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
