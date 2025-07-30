#!/bin/bash

data_dir=

hubert_path=
hubert_layer=
km_path=

nj=16

. local/parse_options.sh

if [ -z "${data_dir}" ]; then
    echo "Please set the data directory using --data_dir"
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


echo "Extracting speaker embeddings"
for subset in train valid; do
    if [ -e ${data_dir}/${subset}.spkemb.list ]; then
        echo "SpkEmb-involving manifest for ${subset} already exists, skipping extraction."
    else
        python -m torch.distributed.run --nproc_per_node=${nj} \
            local/extract_speaker_embedding.py \
                --manifest ${data_dir}/${subset}.raw.list \
                --output ${data_dir}/spkemb/${subset}_spkemb \
                --output_manifest ${data_dir}/${subset}.spkemb.list
    fi
done

echo "Extracting HuBERT tokens"
for subset in train valid; do
    python -m torch.distributed.run --nproc_per_node=${nj} \
        local/extract_hubert_tokens.py \
            --manifest ${data_dir}/${subset}.spkemb.list \
            --output_manifest ${data_dir}/${subset}.list \
            --dense_model ${hubert_path} \
            --layer ${hubert_layer} \
            --quantizer_model ${km_path}
done
