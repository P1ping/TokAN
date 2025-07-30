# Code mainly copied from fairseq_cli/generate.py
"""
Convert extracted data with a trained model.
"""

import ast
import logging
import os
import sys
from argparse import Namespace

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, scoring, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

from fairseq_cli.generate import get_symbols_to_strip_from_output


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"
    assert cfg.common_eval.results_path is not None, "results_path is required to dump manifest"

    os.makedirs(cfg.common_eval.results_path, exist_ok=True)
    manifest_path = os.path.join(cfg.common_eval.results_path, "generate-{}.tsv".format(cfg.dataset.gen_subset))

    # Load original manifest to get headers and data
    original_manifest_path = os.path.join(cfg.task.data, f"{cfg.dataset.gen_subset}.tsv")
    original_manifest = {}
    with open(original_manifest_path) as f:
        header_line = f.readline().strip()
        original_headers = header_line.split("\t")
        assert "gen_tokens" not in original_headers, "gen_tokens column already exists in input manifest"
        output_headers = original_headers + ["gen_tokens"]

        # Load all original data
        for header in original_headers:
            original_manifest[header] = []

        for line in f:
            values = line.strip().split("\t")
            for i, header in enumerate(original_headers):
                original_manifest[header].append(values[i])

    with open(manifest_path, "w", buffering=1, encoding="utf-8") as f_manifest:
        f_manifest.write("\t".join(output_headers) + "\n")
        return dump_manifest(cfg, f_manifest, original_manifest, original_headers)


def dump_manifest(cfg: DictConfig, output_file, original_manifest, original_headers):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("examples.speech_token_to_token.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(cfg.common_eval.path),
        task=None,
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    model = models[0].cuda() if use_cuda else models[0]

    # Set dictionaries
    src_dict = getattr(task, "source_dictionary", None)
    tgt_dict = task.target_dictionary

    # Load dataset splits
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Optimize ensemble for generation
    for model in models:
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(task.max_positions(), *[m.max_positions() for m in models]),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=1,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    generator = task.build_generator(models, cfg.generation)

    def decode_fn(x):
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_utterances = 0
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            assert sample["target"] is not None

            # Remove padding
            src_tokens = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], src_dict.pad())
            target_tokens = utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()

            src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
            target_str = tgt_dict.string(
                target_tokens,
                cfg.common_eval.post_process,
                escape_unk=True,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )

            src_str = decode_fn(src_str)
            target_str = decode_fn(target_str)

            # Process the top-1 prediction
            hypo = hypos[i][0]
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=hypo["alignment"],
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=cfg.common_eval.post_process,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            )
            detok_hypo_str = decode_fn(hypo_str)

            # Score only the top hypothesis
            if cfg.common_eval.post_process is not None:
                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                hypo_tokens = tgt_dict.encode_line(detok_hypo_str, add_if_not_exist=True)
            if hasattr(scorer, "add_string"):
                scorer.add_string(target_str, detok_hypo_str)
            else:
                scorer.add(target_tokens, hypo_tokens)

            # Print to manifest
            utt_id = sample["utt_ids"][i]

            # Find the original row data by matching utt_id
            original_row_idx = original_manifest["id"].index(utt_id)
            original_row = []
            for header in original_headers:
                original_row.append(original_manifest[header][original_row_idx])

            # Append generated tokens
            output_row = original_row + [detok_hypo_str]
            manifest_line = "\t".join(output_row)
            print(manifest_line, file=output_file)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_utterances += sample["nutterances"] if "nutterances" in sample else sample["id"].numel()

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Converted {:,} utterances ({:,} tokens) in {:.1f}s ({:.2f} utterances/s, {:.2f} tokens/s)".format(
            num_utterances,
            gen_timer.n,
            gen_timer.sum,
            num_utterances / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    print(
        "Generate {} with beam={}: {}".format(cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()),
        file=sys.stdout,
    )

    return scorer


if __name__ == "__main__":
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
