#!/bin/bash

ft_data_dir=
pt_data_dir=

hubert_path=
hubert_layer=
km_path=

nj=16

. local/parse_options.sh

if [ -z "${ft_data_dir}" ]; then
    echo "Please set the pre-training data directory using --ft_data_dir"
    exit 1
fi
if [ -z "${pt_data_dir}" ]; then
    echo "Please set the fine-tuning data directory using --pt_data_dir"
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

echo "Extracting HuBERT tokens -> ${ft_data_dir}/[train|valid|test].tokens.tsv"
for subset in train valid test; do
    python -m torch.distributed.run --nproc_per_node=${nj} \
        local/extract_hubert_tokens.py \
            --manifest ${ft_data_dir}/${subset}.raw.tsv \
            --output ${ft_data_dir}/${subset}.tokens.tsv \
            --dense_model ${hubert_path} \
            --layer ${hubert_layer} \
            --quantizer_model ${km_path} \
            --deduplicate
done

echo "Extracting accent embeddings -> ${ft_data_dir}/[train|valid|test].accent.tsv"
for subset in train valid test; do
    python -m torch.distributed.run --nproc_per_node=${nj} \
        local/extract_accent_embedding.py \
            --manifest ${ft_data_dir}/${subset}.tokens.tsv \
            --output ${ft_data_dir}/accent_embeddings/${subset} \
            --output_manifest ${ft_data_dir}/${subset}.accent.tsv
done

echo "Making auxiliary phonemized text and dictionaries -> ${ft_data_dir}/[train|valid].tsv"
# NOTE: The dictionaries should be leveraged from the pre-training data.
# Therefore, we do not generate dictionaries here.
for subset in train valid test; do
    python local/prepare_fairseq_manifest.py \
        --manifest ${ft_data_dir}/${subset}.accent.tsv \
        --output_manifest ${ft_data_dir}/${subset}.tsv \
        --gen_aux_text
done

echo "Copying dictionaries from pre-training data directory to fine-tuning data directory"
for dict_tag in src tgt aux; do
    if [ ! -f ${pt_data_dir}/dict.${dict_tag}.txt ]; then
        echo "Error: Dictionary ${pt_data_dir}/dict.${dict_tag}.txt does not exist."
        exit 1
    fi
    cp ${pt_data_dir}/dict.${dict_tag}.txt ${ft_data_dir}/dict.${dict_tag}.txt
done
