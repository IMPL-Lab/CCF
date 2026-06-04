config=projects/configs/ccf/eval/ccf_source
weight=checkpoints/eval/ccf_source.pth
log_dir=log/ccf

GPUS=${GPUS:-1}

mkdir -p ${log_dir}

bash tools/dist_test.sh ${config}-night.py \
                        ${weight} \
                        ${GPUS} \
                        --eval bbox \
                        > ${log_dir}/night.log 2>&1

bash tools/dist_test.sh ${config}-rain.py \
                        ${weight} \
                        ${GPUS} \
                        --eval bbox \
                        > ${log_dir}/rain.log 2>&1

bash tools/dist_test.sh ${config}-boston.py \
                        ${weight} \
                        ${GPUS} \
                        --eval bbox \
                        > ${log_dir}/boston.log 2>&1
