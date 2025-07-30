#!/bin/bash

data_dir=
nj=16

. local/parse_options.sh

if [ -z "${data_dir}" ]; then
    echo "Please set the data directory using --data_dir"
    exit 1
fi

for subset in train valid; do
    python -m torch.distributed.run --nproc_per_node=${nj} \
        local/extract_speaker_embedding.py \
            --manifest ${data_dir}/${subset}.raw.list \
            --output ${data_dir}/spkemb/${subset}_spkemb \
            --output_manifest ${data_dir}/${subset}.list
done
