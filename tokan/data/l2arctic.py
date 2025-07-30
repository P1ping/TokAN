# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

# This files stores some basic information about the L2-ARCTIC dataset,
# as well as the subset partitioning results in the paper.

import os
from collections import OrderedDict

ACCENT_TO_SPEAKER = {
    "<ar>": ["ABA", "YBAA", "ZHAA", "SKA"],
    "<zh>": ["BWC", "LXC", "NCC", "TXHC"],
    "<hi>": ["ASI", "RRBI", "SVBI", "TNI"],
    "<ko>": ["HJK", "YDCK", "YKWK", "HKK"],
    "<es>": ["EBVS", "ERMS", "NJS", "MBMPS"],
    "<vi>": ["HQTV", "PNV", "THV", "TLV"],
    "<us>": ["BDL", "RMS", "SLT", "CLB"],
}
SPEAKER_TO_ACCENT = {spk: accent for accent, spks in ACCENT_TO_SPEAKER.items() for spk in spks}

SPEAKERS = list(SPEAKER_TO_ACCENT.keys())
UNSEEN_SPEAKERS = ["SKA", "TXHC", "TNI", "HKK", "MBMPS", "TLV", "CLB"]
SEEN_SPEAKERS = [spk for spk in SPEAKERS if spk not in UNSEEN_SPEAKERS]

NUM_A_SENTENCES = 593
NUM_B_SENTENCES = 539

VALID_SENTENCES = {
    "arctic_a0012",
    "arctic_a0038",
    "arctic_a0114",
    "arctic_a0135",
    "arctic_a0176",
    "arctic_a0213",
    "arctic_a0245",
    "arctic_a0282",
    "arctic_a0314",
    "arctic_a0327",
    "arctic_a0329",
    "arctic_a0342",
    "arctic_a0346",
    "arctic_a0375",
    "arctic_a0450",
    "arctic_a0460",
    "arctic_a0484",
    "arctic_a0500",
    "arctic_a0501",
    "arctic_a0503",
    "arctic_a0513",
    "arctic_a0570",
    "arctic_a0575",
    "arctic_a0578",
    "arctic_b0011",
    "arctic_b0041",
    "arctic_b0053",
    "arctic_b0059",
    "arctic_b0105",
    "arctic_b0112",
    "arctic_b0116",
    "arctic_b0188",
    "arctic_b0193",
    "arctic_b0194",
    "arctic_b0204",
    "arctic_b0216",
    "arctic_b0237",
    "arctic_b0245",
    "arctic_b0310",
    "arctic_b0311",
    "arctic_b0324",
    "arctic_b0385",
    "arctic_b0388",
    "arctic_b0392",
    "arctic_b0398",
    "arctic_b0409",
    "arctic_b0478",
    "arctic_b0491",
    "arctic_b0496",
    "arctic_b0533",
}
TEST_SENTENCES = {
    "arctic_a0017",
    "arctic_a0026",
    "arctic_a0045",
    "arctic_a0046",
    "arctic_a0053",
    "arctic_a0059",
    "arctic_a0065",
    "arctic_a0069",
    "arctic_a0084",
    "arctic_a0088",
    "arctic_a0103",
    "arctic_a0109",
    "arctic_a0154",
    "arctic_a0175",
    "arctic_a0199",
    "arctic_a0209",
    "arctic_a0216",
    "arctic_a0225",
    "arctic_a0233",
    "arctic_a0252",
    "arctic_a0268",
    "arctic_a0290",
    "arctic_a0296",
    "arctic_a0300",
    "arctic_a0312",
    "arctic_a0354",
    "arctic_a0367",
    "arctic_a0368",
    "arctic_a0372",
    "arctic_a0374",
    "arctic_a0389",
    "arctic_a0391",
    "arctic_a0396",
    "arctic_a0424",
    "arctic_a0436",
    "arctic_a0463",
    "arctic_a0521",
    "arctic_a0543",
    "arctic_a0551",
    "arctic_a0554",
    "arctic_a0589",
    "arctic_b0004",
    "arctic_b0025",
    "arctic_b0029",
    "arctic_b0031",
    "arctic_b0032",
    "arctic_b0037",
    "arctic_b0038",
    "arctic_b0048",
    "arctic_b0049",
    "arctic_b0066",
    "arctic_b0087",
    "arctic_b0095",
    "arctic_b0098",
    "arctic_b0107",
    "arctic_b0128",
    "arctic_b0140",
    "arctic_b0141",
    "arctic_b0145",
    "arctic_b0154",
    "arctic_b0223",
    "arctic_b0238",
    "arctic_b0299",
    "arctic_b0343",
    "arctic_b0344",
    "arctic_b0346",
    "arctic_b0353",
    "arctic_b0375",
    "arctic_b0383",
    "arctic_b0407",
    "arctic_b0419",
    "arctic_b0421",
    "arctic_b0434",
    "arctic_b0439",
    "arctic_b0441",
    "arctic_b0450",
    "arctic_b0487",
    "arctic_b0490",
    "arctic_b0503",
    "arctic_b0515",
}


def get_sentence_to_subset():
    sid2subset = {}

    for idx in range(1, NUM_A_SENTENCES + 1):
        sid = "arctic_a{}".format(str(idx).zfill(4))
        if sid in TEST_SENTENCES:
            sid2subset[sid] = "test"
        elif sid in VALID_SENTENCES:
            sid2subset[sid] = "valid"
        else:
            sid2subset[sid] = "train"

    for idx in range(1, NUM_B_SENTENCES + 1):
        sid = "arctic_b{}".format(str(idx).zfill(4))
        if sid in TEST_SENTENCES:
            sid2subset[sid] = "test"
        elif sid in VALID_SENTENCES:
            sid2subset[sid] = "valid"
        else:
            sid2subset[sid] = "train"

    return sid2subset


def load_metadata(data_dir):
    text_path = os.path.join(data_dir, "PROMPTS")
    sid2text = OrderedDict()
    with open(text_path, "r") as f:
        for line in f:
            item = line.strip().lstrip("(").rstrip(")").strip()
            sid, text = item.split(" ", 1)
            sid2text[sid] = eval(text)  # Remove the quotes

    metadata = []
    for spk in SPEAKER_TO_ACCENT.keys():
        spk_dir = os.path.join(data_dir, spk)

        # Skip if speaker directory doesn't exist
        if not os.path.exists(spk_dir):
            print(f"Warning: Speaker directory {spk_dir} not found, skipping.")
            continue

        wav_dir = os.path.join(spk_dir, "wav")
        transcript_dir = os.path.join(spk_dir, "transcript")

        # Make sure needed directories exist
        if not os.path.exists(wav_dir):
            print(f"Warning: Required wav directories for speaker {spk} not found, skipping.")
            continue
        if not os.path.exists(transcript_dir):
            print(f"Warning: Required transcript directories for speaker {spk} not found, will use PROMPTS instead.")

        # Iterate through WAV files
        for wav_file in os.listdir(wav_dir):
            if not wav_file.endswith(".wav"):
                continue

            sid = os.path.splitext(wav_file)[0]
            wav_path = os.path.join(wav_dir, wav_file)
            text_path = os.path.join(transcript_dir, f"{sid}.txt")

            # Read transcript
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                text = sid2text[sid]
            metadata.append((spk, sid, text, wav_path))

    return metadata
