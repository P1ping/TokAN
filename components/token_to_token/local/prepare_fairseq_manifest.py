import argparse
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict, defaultdict

from tokan.utils import get_pylogger
from tokan.utils.phoneme_tokenizer import PhonemeTokenizer

logger = get_pylogger(__name__)


MANIFEST_COLUMNS = [
    "id",
    "src_audio",
    "src_tokens",
    "src_n_tokens",
    "tgt_audio",
    "tgt_tokens",
    "tgt_n_tokens",
    "src_embed",  # optional
    "aux_text",  # optional
]


def deduplicate_units(units):
    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out


def load_units(in_file, deduplicate):
    out_wav_paths = []
    out_units = []
    token2freq = defaultdict(int)
    with open(in_file, encoding="utf-8") as f:
        for line in f:
            wav_path, unit_str = line.strip().split("\t")
            wav_path = Path(wav_path).as_posix()
            units = unit_str.split(" ")
            if deduplicate:
                units = deduplicate_units(units)
            for u in units:
                token2freq[u] += 1
            out_wav_paths.append(wav_path)
            out_units.append(units)
    return out_wav_paths, out_units, token2freq


def load_scp_emb(scp_file):
    wav_paths = []
    emb_paths = []
    with open(scp_file) as f:
        for line in f:
            wav_path, emb_path = line.strip().split()
            wav_path = Path(wav_path).as_posix()
            wav_paths.append(wav_path)
            emb_paths.append(emb_path)
    return wav_paths, emb_paths


def load_dict_file(dict_file):
    token2freq = OrderedDict()
    with open(dict_file) as f:
        for i, line in enumerate(f):
            token, freq = line.strip().split()
            token2freq[token] = int(freq)
    return token2freq


def write_dict_file(token2freq, out_file, sort=True):
    if sort:
        token2freq_ordered = sorted(token2freq.items(), key=lambda x: x[1], reverse=True)
    else:
        token2freq_ordered = token2freq.items()
    with open(out_file, "w") as f:
        for token, freq in token2freq_ordered:
            f.write(f"{token} {freq}\n")


def process(args):
    output_root = Path(args.output_manifest).parent
    output_root.mkdir(exist_ok=True)

    if args.gen_aux_text:
        phonemizer = PhonemeTokenizer(trace_freq=True)

    # Load the input manifest for the specific split
    with open(args.manifest, "r", encoding="utf-8") as f_in:
        header_line = next(f_in).strip()
        header = header_line.split("\t")

        # Get column indices
        src_tokens_idx = header.index("src_tokens") if "src_tokens" in header else None
        tgt_tokens_idx = header.index("tgt_tokens") if "tgt_tokens" in header else None
        text_idx = header.index("text") if "text" in header else None

        # Check if token count columns exist
        src_n_tokens_idx = header.index("src_n_tokens") if "src_n_tokens" in header else None
        tgt_n_tokens_idx = header.index("tgt_n_tokens") if "tgt_n_tokens" in header else None

        # Read all rows
        rows = []
        for line in f_in:
            row = line.strip().split("\t")
            rows.append(row)

    print(f"Loaded manifest with {len(rows)} rows and columns: {header}")

    # Initialize token frequency counters
    src_token2freq = defaultdict(int)
    tgt_token2freq = defaultdict(int)

    print("Processing manifest...")

    # Process each row for token frequency counting
    if args.gen_dict:
        for row in tqdm(rows, desc="Counting token frequencies"):
            if src_tokens_idx is not None and row[src_tokens_idx].strip():
                src_tokens = row[src_tokens_idx].split()
                for token in src_tokens:
                    src_token2freq[token] += 1

            if tgt_tokens_idx is not None and row[tgt_tokens_idx].strip():
                tgt_tokens = row[tgt_tokens_idx].split()
                for token in tgt_tokens:
                    tgt_token2freq[token] += 1

    # Add phonemized text if requested
    aux_text_list = []
    if args.gen_aux_text and text_idx is not None:
        for row in tqdm(rows, desc="Phonemizing text"):
            if row[text_idx].strip():
                aux_phonemes = phonemizer.phonemize(row[text_idx])
                aux_text = " ".join(aux_phonemes)
            else:
                aux_text = ""
            aux_text_list.append(aux_text)

    # Prepare output header
    output_header = header[:]
    if args.gen_aux_text and text_idx is not None and "aux_text" not in header:
        output_header.append("aux_text")

    # Add token count columns if they don't exist
    if src_tokens_idx is not None and src_n_tokens_idx is None:
        output_header.append("src_n_tokens")
    if tgt_tokens_idx is not None and tgt_n_tokens_idx is None:
        output_header.append("tgt_n_tokens")

    # Save the processed manifest
    print(f"Writing manifest to {args.output_manifest}...")
    with open(args.output_manifest, "w", encoding="utf-8") as f_out:
        # Write header
        f_out.write("\t".join(output_header) + "\n")

        # Write data rows
        for i, row in enumerate(rows):
            output_row = row[:]

            # Add aux_text if requested
            if args.gen_aux_text and text_idx is not None and "aux_text" not in header:
                output_row.append(aux_text_list[i])

            # Add token counts if they don't exist
            if src_tokens_idx is not None and src_n_tokens_idx is None:
                src_n_tokens = len(row[src_tokens_idx].split()) if row[src_tokens_idx].strip() else 0
                output_row.append(str(src_n_tokens))

            if tgt_tokens_idx is not None and tgt_n_tokens_idx is None:
                tgt_n_tokens = len(row[tgt_tokens_idx].split()) if row[tgt_tokens_idx].strip() else 0
                output_row.append(str(tgt_n_tokens))

            f_out.write("\t".join(output_row) + "\n")

    # Generate dictionaries only if requested (typically for training set)
    if args.gen_dict:
        print("Generating dictionaries...")

        # Handle unified dictionary option
        if args.unify_dict:
            print("Creating unified dictionary for source and target tokens...")
            # Merge source and target token frequencies
            unified_token2freq = defaultdict(int)
            for token, freq in src_token2freq.items():
                unified_token2freq[token] += freq
            for token, freq in tgt_token2freq.items():
                unified_token2freq[token] += freq

            # Use unified dictionary for both source and target
            src_token2freq = unified_token2freq
            tgt_token2freq = unified_token2freq

        output_root = Path(args.output_manifest).parent
        # Write source dictionary
        src_dict_path = output_root / "dict.src.txt"
        write_dict_file(src_token2freq, src_dict_path)
        print(f"Wrote source dictionary to {src_dict_path} ({len(src_token2freq)} tokens)")

        # Write target dictionary
        tgt_dict_path = output_root / "dict.tgt.txt"
        write_dict_file(tgt_token2freq, tgt_dict_path)
        print(f"Wrote target dictionary to {tgt_dict_path} ({len(tgt_token2freq)} tokens)")

        # Write auxiliary text dictionary if phonemization was performed
        if args.gen_aux_text:
            aux_dict_path = output_root / "dict.aux.txt"
            write_dict_file(phonemizer.dict, aux_dict_path)
            print(f"Wrote auxiliary text dictionary to {aux_dict_path} ({len(phonemizer.dict)} tokens)")

    print("Processing completed!")


def get_parser():
    parser = argparse.ArgumentParser(description="Process individual TSV manifest files for fairseq training.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the input manifest file with columns: id, src_audio, tgt_audio, text, ...",
    )
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to the output manifest file")

    parser.add_argument("--gen_aux_text", action="store_true", help="Generate auxiliary phonemized text")

    parser.add_argument("--gen_dict", action="store_true", help="Generate dictionaries (typically for training set)")
    parser.add_argument(
        "--unify_dict", action="store_true", help="Create unified dictionary for source and target tokens"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    process(args)
