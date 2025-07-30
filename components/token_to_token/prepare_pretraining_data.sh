#!/bin/bash

pt_data_dir=

hubert_path=
hubert_layer=
km_path=

nj=16

. local/parse_options.sh

if [ -z "${pt_data_dir}" ]; then
    echo "Please set the pre-training data directory using --pt_data_dir"
    exit 1
fi
if [ -z "${hubert_path}" ]; then
    echo "Please set the HuBERT model path using --hubert_path"
    exit 1
fi
if [ -z "${hubert_layer}" ]; then
    echo "Please set the HuBERT layer using --hubert_layer"
    exit 1
fi  
if [ -z "${km_path}" ]; then
    echo "Please set the quantizer model path using --km_path"
    exit 1
fi

echo "Extracting HuBERT tokens -> ${pt_data_dir}/[train|valid].tokens.tsv"
for subset in train valid; do
    python -m torch.distributed.run --nproc_per_node=${nj} \
        local/extract_hubert_tokens.py \
            --manifest ${pt_data_dir}/${subset}.raw.tsv \
            --output ${pt_data_dir}/${subset}.tokens.tsv \
            --dense_model ${hubert_path} \
            --layer ${hubert_layer} \
            --quantizer_model ${km_path} \
            --deduplicate \
            --use_src_as_tgt
done

# # NOTE: Accent embeddings are used in fine-tuning, not in pre-training. So, skip this step for now.
# echo "Extracting accent embeddings -> ${pt_data_dir}/[train|valid].accent.tsv"
# for subset in train valid; do
#     python -m torch.distributed.run --nproc_per_node=${nj} \
#         local/extract_accent_embedding.py \
#             --manifest ${pt_data_dir}/${subset}.tokens.tsv \
#             --output ${pt_data_dir}/accent_embeddings/${subset} \
#             --output_manifest ${pt_data_dir}/${subset}.accent.tsv
# done

echo "Making auxiliary phonemized text and dictionaries -> ${pt_data_dir}/[train|valid].tsv"
# NOTE: The dictionaries are generated from the training set only
python local/prepare_fairseq_manifest.py \
    --manifest ${pt_data_dir}/train.tokens.tsv \
    --output_manifest ${pt_data_dir}/train.tsv \
    --gen_aux_text --gen_dict --unify_dict
python local/prepare_fairseq_manifest.py \
    --manifest ${pt_data_dir}/valid.tokens.tsv \
    --output_manifest ${pt_data_dir}/valid.tsv \
    --gen_aux_text
