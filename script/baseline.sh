export CUDA_VISIBLE_DEVICES=7
all_language_pairs="zh-en" # zh-en en-zh de-en en-de cs-en en-cs ru-en en-ru
cmp_models="alma7b alpaca-7b vicuna nllb-3.3b bayling-13b ALMA-13B-LoRA Bigtranslate13B gpt3.5-text-davinci-003 GPT-4-1106-preview" #alma7b alpaca-7b vicuna nllb-3.3b bayling-13b ALMA-13B-LoRA Bigtranslate13B gpt3.5-text-davinci-003 GPT-4-1106-preview
few_shot="zero-shot"
num_test="all"

for test_pair in $all_language_pairs
do
    for cmp_model in $cmp_models    
    do
        export TEST_PAIRS=$test_pair
        export TEST_FILE_PATH="../dataset/test/${few_shot}/${TEST_PAIRS}/${num_test}/${test_pair}_${cmp_model}.json"
        export SAVE_DIR="../results/baseline/${TEST_PAIRS}/${num_test}_results"
        mkdir -p "$(dirname "$SAVE_DIR")"

        echo "Testing ${TEST_FILE_PATH}"
        echo "Comparing with ${cmp_model}"
        echo "Testing pair ${TEST_PAIRS}"
        echo "Testing z/f-shot ${few_shot}"

        src=$(echo "${TEST_PAIRS}" | cut -d "-" -f 1)
        tgt=$(echo "${TEST_PAIRS}" | cut -d "-" -f 2)

        if [ "${tgt}" = "zh" ]; then
            TOK="zh"
        else
            TOK="13a"
        fi

        echo "--------------------Results for ${TEST_PAIRS}--${cmp_model}----------------------"
        src_path="../dataset/test/${few_shot}/${TEST_PAIRS}/${num_test}/all_source.txt"
        tgt_path="../dataset/test/${few_shot}/${TEST_PAIRS}/${num_test}/all_target.txt"
        json_path="$TEST_FILE_PATH"
        output_path="../results/baseline/${TEST_PAIRS}/${num_test}_results/${cmp_model}"

        mkdir -p "$(dirname "$output_path")"

        jq -r '.[] | .translation.medium' "${json_path}" > "${output_path}.txt"

        SACREBLEU_FORMAT=text sacrebleu -tok "${TOK}" -w 2 "${tgt_path}" < "${output_path}.txt" > "${output_path}.bleu"
        cat "${output_path}.bleu"

        comet-score -s "${src_path}" -t "${output_path}.txt" -r "${tgt_path}" --batch_size 64 --model /data/zhaopengfeng/download_model/Unbabel/wmt22-comet-da/checkpoints/model.ckpt --gpus 1 > "${output_path}.comet"
        comet-score -s "${src_path}" -t "${output_path}.txt" --batch_size 64 --model /data/zhaopengfeng/download_model/Unbabel/wmt22-cometkiwi-da/checkpoints/model.ckpt --gpus 1 > "${output_path}.cometkiwi"

        echo "-------------------------${cmp_model}--${src}-${tgt}-----------------------------"
        cat "${output_path}.bleu"
        tail -n 1 "${output_path}.comet"
        tail -n 1 "${output_path}.cometkiwi"

    done
    python ../read_score.py "${SAVE_DIR}"
done
