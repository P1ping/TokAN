#!/bin/bash

stage=-1
stop_stage=-1

libritts_root=
l2arctic_root=

hubert_path=
hubert_layer=17
n_clusters=1000
km_path=

text_to_mel_ckpt=    # Should be specified before synthetic data generation
token_to_token_ckpt=    # Should be specified before evaluation
token_to_mel_ckpt=    # Should be specified before evaluation

bigvgan_tag_or_ckpt="nvidia/bigvgan_22khz_80band"  # Or path to the checkpoint with config.json in the same directory

nj=8    # Number of jobs for parallel processing

. scripts/parse_options.sh || exit 1;


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Data preparation"

    echo "Verifying data directories and completeness..."
    python scripts/verify_data_preparation.py \
        --libritts_root ${libritts_root} \
        --l2arctic_root ${l2arctic_root}

    if [ $? -ne 0 ]; then
        echo "❌ Data verification failed. Please check the output above for guidance."
        echo "The verification script provides detailed instructions for downloading missing datasets."
        exit 1
    fi

    echo "✅ Data verification completed successfully!"
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: K-means model check"
    echo "If you have changed the number of clusters, please also update n_vocab in components/token_to_mel/configs/model/yirga.yaml"

    echo "Checking if HuBERT model exists..."
    if [ ! -f ${hubert_path} ]; then
        echo "❌ HuBERT model not found at: ${hubert_path}"
    else
        echo "✅ HuBERT model found: ${hubert_path}"
    fi

    echo "Checking if K-means model exists..."
    if [ ! -f ${km_path} ]; then
        echo "⚠️  K-means model not found at: ${km_path}"
        echo "Expected clusters: ${n_clusters}, HuBERT layer: ${hubert_layer}"
    else
        echo "✅ K-means model found: ${km_path}"
        echo "Clusters: ${n_clusters}, HuBERT layer: ${hubert_layer}"
    fi

    if [ ! -f ${hubert_path} ] || [ ! -f ${km_path} ]; then
        echo "❗️ HuBERT or K-means model not found."
        echo "You can download the pretrained one by running tokan/utils/model_utils.py (see README.md for details) and specify the paths."
        echo "Alternatively, you can manually train a K-means model (see third_party/fairseq/examples/hubert/simple_kmeans/README.md)."
    else
        echo "✅ HuBERT and K-means models are ready."
    fi
fi


text_to_mel_data_dir=$(pwd)/components/text_to_mel/data/libritts
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: TTS training for synthetic data generation"

    echo "Generating manifests for LibriTTS -> ${text_to_mel_data_dir}/[train|valid].raw.list"
    python components/text_to_mel/local/collect_libritts.py \
        --libritts_root ${libritts_root} \
        --output_dir ${text_to_mel_data_dir}

    cur_dir=$(pwd)
    cd components/text_to_mel || exit 1
    echo "Extracting speaker embeddings"
    bash prepare_data.sh \
        --data_dir ${text_to_mel_data_dir} \
        --nj ${nj}
    cd ${cur_dir} || exit 1

    echo "Training TTS model"
    echo "You can change the configurations (e.g., batch_size) in components/text_to_mel/configs/"
    echo "Checkpoints will be saved in components/text_to_mel/experiments"
    python components/text_to_mel/train.py \
        experiment=libritts \
        data.train_filelist_path=${text_to_mel_data_dir}/train.list \
        data.valid_filelist_path=${text_to_mel_data_dir}/valid.list
fi


token_to_mel_data_dir=$(pwd)/components/token_to_mel/data/libritts
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Token-to-Mel synthesizer training"

    echo "Generating manifests for LibriTTS -> ${token_to_mel_data_dir}/[train|valid].raw.list"
    if [ -e ${text_to_mel_data_dir}/train.list ] && [ -e ${text_to_mel_data_dir}/valid.list ]; then
        echo "Using existing manifests from text_to_mel_data_dir"
        mkdir -p ${token_to_mel_data_dir}
        cp ${text_to_mel_data_dir}/train.list ${token_to_mel_data_dir}/train.spkemb.list
        cp ${text_to_mel_data_dir}/valid.list ${token_to_mel_data_dir}/valid.spkemb.list
    else
        echo "Generating manifests for LibriTTS -> ${token_to_mel_data_dir}/[train|valid].raw.list"
        python components/token_to_mel/local/collect_libritts.py \
            --libritts_root ${libritts_root} \
            --output_dir ${token_to_mel_data_dir}
    fi

    cur_dir=$(pwd)
    cd components/token_to_mel || exit 1
    echo "Extracting HuBERT tokens and speaker embeddings"
    bash prepare_data.sh \
        --data_dir ${token_to_mel_data_dir} \
        --nj ${nj} \
        --hubert_path ${hubert_path} \
        --hubert_layer ${hubert_layer} \
        --km_path ${km_path}
    cd ${cur_dir} || exit 1

    echo "Training token-to-Mel model"
    echo "You can change the configurations (e.g., batch_size) in components/token_to_mel/configs/"
    echo "By default, the duration predictor is regression-based, you can switch to flow matching"
    echo " by modifying components/token_to_mel/configs/model/encoder/default.yaml"
    echo "Checkpoints will be saved in components/token_to_mel/experiments"
    python components/token_to_mel/train.py \
        experiment=libritts_token \
        data.train_metadata_path=${token_to_mel_data_dir}/train.list \
        data.valid_metadata_path=${token_to_mel_data_dir}/valid.list
fi


pt_data_dir=$(pwd)/components/token_to_token/data/libritts
pt_exp_dir=$(pwd)/components/token_to_token/experiments/libritts_pretrain
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Token-to-token conversion pre-training"

    echo "Generating manifests for LibriTTS -> ${pt_data_dir}/[train|valid].raw.tsv"
    python $(pwd)/components/token_to_token/local/collect_libritts.py \
        --libritts_root ${libritts_root} \
        --output_dir ${pt_data_dir}

    cur_dir=$(pwd)
    cd components/token_to_token || exit 1
    bash prepare_pretraining_data.sh \
        --pt_data_dir ${pt_data_dir} \
        --nj ${nj} \
        --hubert_path ${hubert_path} \
        --hubert_layer ${hubert_layer} \
        --km_path ${km_path}
    cd ${cur_dir} || exit 1
    
    echo "Pre-training the token-to-token conversion model..."
    fairseq-train ${pt_data_dir} \
        --user-dir tokan/fairseq_modules \
        --train-subset train --valid-subset valid \
        --task speech_token_denoising \
        --source-lang src \
        --mask-ratio 0.3 --poisson-lambda 3.0 \
        --random-ratio 0.1 \
        --insert-ratio 0.1 --replace-length 1 \
        --arch transformer_base++ \
        --share-all-embeddings \
        --ddp-backend legacy_ddp --fp16 \
        --criterion label_smoothed_cross_entropy_with_ctc \
        --ctc-weight 1.0 --load-aux-text \
        --label-smoothing 0.1 \
        --optimizer adam \
        --adam-betas '(0.9, 0.999)' \
        --adam-eps 1e-08 \
        --clip-norm 0.1 \
        --lr-scheduler polynomial_decay \
        --lr 1e-4 \
        --warmup-updates 1000 \
        --total-num-update 600000 \
        --dropout 0.1 \
        --attention-dropout 0.1 \
        --weight-decay 0.01 \
        --max-tokens 15000 \
        --update-freq 2 \
        --skip-invalid-size-inputs-valid-test \
        --save-interval-updates 10000 --no-epoch-checkpoints \
        --log-format simple \
        --log-interval 100 \
        --save-dir ${pt_exp_dir} \
        --tensorboard-logdir ${pt_exp_dir}/tensorboard
fi


ft_data_dir=$(pwd)/components/token_to_token/data/l2arctic
ft_exp_dir=$(pwd)/components/token_to_token/experiments/l2arctic_finetune
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Stage 5: Token-to-token conversion fine-tuning"

    if [ -z ${text_to_mel_ckpt} ]; then
        echo "Please specify the text-to-Mel checkpoint with --text_to_mel_ckpt"
        exit 1
    fi
    echo "Synthesizing L1-accented target data for L2ARCTIC -> ${ft_data_dir}/synthetic_target"
    echo "Generating manifests for L2ARCTIC -> ${ft_data_dir}/[train|valid|test].raw.tsv"
    python -m torch.distributed.run --nproc_per_node=${nj} \
        scripts/generation/generate_l2arctic_target.py \
            --data_dir ${l2arctic_root} \
            --matcha_ckpt ${text_to_mel_ckpt} \
            --output_wav_dir ${ft_data_dir}/synthetic_target \
            --output_manifest_dir ${ft_data_dir} \
            --bigvgan_tag_or_ckpt ${bigvgan_tag_or_ckpt}

    cur_dir=$(pwd)
    cd components/token_to_token || exit 1
    bash prepare_finetuning_data.sh \
        --ft_data_dir ${ft_data_dir} \
        --pt_data_dir ${pt_data_dir} \
        --nj ${nj} \
        --hubert_path ${hubert_path} \
        --hubert_layer ${hubert_layer} \
        --km_path ${km_path}
    cd ${cur_dir} || exit 1

    echo "Fine-tuning the token-to-token conversion model..."
    fairseq-train \
        ${ft_data_dir} \
        --source-lang src --target-lang tgt \
        --user-dir tokan/fairseq_modules \
        --task speech_token_to_token \
        --pretrained-checkpoint ${pt_exp_dir}/checkpoint_last.pt \
        --arch transformer_base++ \
        --share-all-embeddings \
        --condition-dim 1024 --load-src-embed \
        --ddp-backend legacy_ddp --fp16 \
        --criterion label_smoothed_cross_entropy_with_ctc \
        --ctc-weight 1.0 --load-aux-text \
        --max-update 100000 \
        --skip-invalid-size-inputs-valid-test \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr-scheduler 'inverse_sqrt' \
        --warmup-init-lr 1e-7  --warmup-updates 2000 \
        --lr 2e-5 --stop-min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001 \
        --dropout 0.3 \
        --label-smoothing 0.1 \
        --max-tokens 20000 \
        --update-freq 1 \
        --patience 20 \
        --save-interval-updates 5000 --no-epoch-checkpoints \
        --save-dir ${ft_exp_dir} \
        --tensorboard-logdir ${ft_exp_dir}/tensorboard
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Stage 6: Generation for evaluation"

    echo "Decoding tokens..."
    if [ -z ${token_to_token_ckpt} ]; then
        token_to_token_ckpt=$(pwd)/components/token_to_token/experiments/l2arctic_finetune/checkpoint_best.pt
        echo "token_to_token_ckpt not specified, using default: ${token_to_token_ckpt}"
        if [ ! -f ${token_to_token_ckpt} ]; then
            echo "Token-to-token checkpoint not found: ${token_to_token_ckpt}"
            echo "Please specify the token-to-token checkpoint with --token_to_token_ckpt"
            exit 1
        fi
    fi
    python scripts/generation/fairseq_decode.py \
        ${ft_data_dir} \
        --user-dir tokan/fairseq_modules \
        --task speech_token_to_token \
        --source-lang src --target-lang tgt \
        --path ${token_to_token_ckpt} \
        --beam 10 \
        --scoring lcsr \
        --results-path ${ft_exp_dir}/test_output

    echo "Generating audio from tokens..."
    if [ -z ${token_to_mel_ckpt} ]; then
        echo "Please specify the token-to-Mel checkpoint with --token_to_mel_ckpt"
        exit 1
    fi
    echo "Please add arguments ---source_duration_scale 1.0 and --force_total_duration if you want to preserve the source total duration"
    python -m torch.distributed.run --nproc_per_node=${nj} \
        scripts/generation/generate_from_fairseq_manifest.py \
            --manifest_path ${ft_exp_dir}/test_output/generate-test.tsv \
            --output_dir ${ft_exp_dir}/test_output/wavs \
            --output_manifest_path ${ft_exp_dir}/test_output/generate-test-wav.tsv \
            --bigvgan_tag_or_ckpt ${bigvgan_tag_or_ckpt} \
            --yirga_ckpt ${token_to_mel_ckpt} \
            # --source_duration_scale 1.0 \
            # --force_total_duration
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "Stage 7: Evaluation"

    echo "Computing WERs..."
    python -m torch.distributed.run --nproc_per_node=${nj} \
        scripts/evaluation/compute_wer.py \
            --manifest_path ${ft_exp_dir}/test_output/generate-test-wav.tsv \
            --output_transcript ${ft_exp_dir}/test_output/recogized_transcript.tsv \
            --output_results ${ft_exp_dir}/test_output/wer_results.txt \
            --tag_to_recognize gen_audio
    
    echo "Computing PPG distances with synthetic targets..."
    python -m torch.distributed.run --nproc_per_node=${nj} \
        scripts/evaluation/compute_ppg_distance.py \
            --manifest_path ${ft_exp_dir}/test_output/generate-test-wav.tsv \
            --output_distances ${ft_exp_dir}/test_output/ppg_distances.tsv \
            --output_results ${ft_exp_dir}/test_output/ppg_distance_results.txt \
            --tag_to_recognize gen_audio

    echo "Computing speaker similarity with source audio..."
    python -m torch.distributed.run --nproc_per_node=${nj} \
        scripts/evaluation/compute_speaker_similarity.py \
            --manifest_path ${ft_exp_dir}/test_output/generate-test-wav.tsv \
            --output_similarities ${ft_exp_dir}/test_output/speaker_similarities.tsv \
            --output_results ${ft_exp_dir}/test_output/speaker_similarity_results.txt
fi
