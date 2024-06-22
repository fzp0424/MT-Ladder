export CUDA_VISIBLE_DEVICES=7
export SEED=424
export BASE_MODEL_PATH="<<<BASE_MODEL_PATH>>>"
export PEFT_MODEL_PATH="<<<PEFT_MODEL_PATH>>>"

all_language_pairs="en-zh" #zh-en en-zh de-en en-de cs-en en-cs ru-en en-ru
cmp_models="alpaca-7b" #Models to refine: alma7b alpaca-7b vicuna nllb-3.3b bayling-13b bayling-7b ALMA-13B-LoRA Bigtranslate13B gpt3.5-text-davinci-003 GPT-4-1106-preview
few_shot="zero-shot" #default
num_test="all" #default

for test_pair in $all_language_pairs
do
    for cmp_model in $cmp_models    
    do
        export PROMPT="intermediate"
        export TEST_PAIRS=$test_pair
        src=$(echo "${TEST_PAIRS}" | cut -d "-" -f 1)
        tgt=$(echo "${TEST_PAIRS}" | cut -d "-" -f 2)
        export TEST_FILE_PATH="../dataset/test/${few_shot}/${TEST_PAIRS}/${num_test}/${test_pair}_${cmp_model}.json"
        export OUTPUT_FILE_PREFIX="${cmp_model}"
        SAVE_DIR="${PEFT_MODEL_PATH}/${num_test}/${test_pair}"  # Directory to save results
        mkdir -p "$(dirname "$SAVE_DIR")"

        echo "Using PEFT model at ${PEFT_MODEL_PATH}"
        echo "Testing ${TEST_FILE_PATH}"
        echo "Comparing with ${cmp_model}"
        echo "Testing pair ${TEST_PAIRS}"
        echo "Save dir ${SAVE_DIR}"

        # run inference.py
        python -u ../inference.py \
        --base-model "${BASE_MODEL_PATH}" \
        --peft-path "${PEFT_MODEL_PATH}" \
        --prompt-strategy "${PROMPT}" \
        --test-pairs "${TEST_PAIRS}" \
        --test-file-path "${TEST_FILE_PATH}" \
        --output-dir "${SAVE_DIR}" \
        --output-file-prefix "${OUTPUT_FILE_PREFIX}" \
        --seed "${SEED}"

        if [ "${tgt}" = "zh" ]; then
            TOK="zh"
        else
            TOK="13a"
        fi

        echo "--------------------Results for ${TEST_PAIRS}--${cmp_model}-------------------------------"
        src_path="../dataset/test/${few_shot}/${TEST_PAIRS}/${num_test}/all_source.txt"
        tgt_path="../dataset/test/${few_shot}/${TEST_PAIRS}/${num_test}/all_target.txt"
        output_path="${SAVE_DIR}/${OUTPUT_FILE_PREFIX}-${src}-${tgt}"
        SACREBLEU_FORMAT=text sacrebleu -tok "${TOK}" -w 2 "${tgt_path}" < "${output_path}" > "${output_path}.bleu"
        cat "${output_path}.bleu"

        comet-score -s "${src_path}" -t "${output_path}" -r "${tgt_path}" --batch_size 64 --model /data/zhaopengfeng/download_model/Unbabel/wmt22-comet-da/checkpoints/model.ckpt --gpus 1 > "${output_path}.comet"
        comet-score -s "${src_path}" -t "${output_path}" --batch_size 64 --model /data/zhaopengfeng/download_model/Unbabel/wmt22-cometkiwi-da/checkpoints/model.ckpt --gpus 1 > "${output_path}.cometkiwi"

        echo "---------------------------${src}-${tgt}-------------------------------"
        cat "${output_path}.bleu"
        tail -n 1 "${output_path}.comet"
        tail -n 1 "${output_path}.cometkiwi"

    done
    python ../read_score.py "${SAVE_DIR}" #Extract scores
done

