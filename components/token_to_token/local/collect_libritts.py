import os
import argparse
import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="LibriTTS dataset preparation")
    parser.add_argument("--libritts_root", type=str, required=True, help="Path to the LibriTTS dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument(
        "--train_subsets",
        nargs="+",
        type=str,
        default=["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other"],
        help="List of training subsets",
    )
    parser.add_argument(
        "--valid_subsets", nargs="+", type=str, default=["test-clean", "test-other"], help="List of validation subsets"
    )
    return parser


def process_subset(subset_path, entries):
    """
    Process a LibriTTS subset, collecting utterance IDs, wav paths, and transcripts
    """
    if not os.path.exists(subset_path):
        print(f"Warning: {subset_path} does not exist, skipping.")
        return

    # Iterate through speaker directories
    for speaker_id in tqdm.tqdm(os.listdir(subset_path), desc=f"Processing {os.path.basename(subset_path)}"):
        speaker_path = os.path.join(subset_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        # Iterate through chapter directories
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue

            # Iterate through utterance files
            for file_name in os.listdir(chapter_path):
                if file_name.endswith(".wav") and not file_name.startswith("."):
                    # Get utt_id without extension
                    utt_id = os.path.splitext(file_name)[0]
                    wav_path = os.path.join(chapter_path, file_name)

                    # Ensure transcript file exists
                    txt_path = os.path.join(chapter_path, utt_id + ".normalized.txt")
                    if not os.path.exists(txt_path):
                        continue

                    # Read transcript
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        text = text.replace("\n", " ").replace("\t", " ")

                    # Add entry with format: utt_id|wav_path|text
                    entries.append((utt_id, os.path.abspath(wav_path), text))


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    train_entries = []
    for subset in args.train_subsets:
        subset_path = os.path.join(args.libritts_root, subset)
        process_subset(subset_path, train_entries)

    valid_entries = []
    for subset in args.valid_subsets:
        subset_path = os.path.join(args.libritts_root, subset)
        process_subset(subset_path, valid_entries)

    # Write train file
    train_file_path = os.path.join(args.output_dir, "train.raw.tsv")
    with open(train_file_path, "w", encoding="utf-8") as f:
        f.write("id\tsrc_audio\ttext\n")  # Header
        for entry in train_entries:
            f.write("\t".join(entry) + "\n")

    # Write valid file
    valid_file_path = os.path.join(args.output_dir, "valid.raw.tsv")
    with open(valid_file_path, "w", encoding="utf-8") as f:
        f.write("id\tsrc_audio\ttext\n")  # Header
        for entry in valid_entries:
            f.write("\t".join(entry) + "\n")

    print(f"Processed {len(train_entries)} (train) and {len(valid_entries)} (valid) utterances")
    print(f"Files created: {train_file_path} and {valid_file_path}")


if __name__ == "__main__":
    main()
