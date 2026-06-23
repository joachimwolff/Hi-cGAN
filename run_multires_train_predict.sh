#!/usr/bin/env bash
#
# Train Hi-cGAN at 5kb / 10kb / 25kb and predict each model at multiple
# target resolutions (1, 2, 5, 10, 25, 50, 100 kb).
#
# Window sizes (must match between training and prediction for a given model):
#   5kb  -> 512 and 768
#   10kb -> 512 and 768
#   25kb -> 256
#
# Run from the Hi-cGAN repo root with the hicgan conda env active:
#   conda activate hicgan
#   ./run_multires_train_predict.sh
#
set -euo pipefail
cd "$(dirname "$0")"            # so the relative BASE path below works

# ---------------------------------------------------------------- config ----
BASE="/mnt/project/wolffjoa/hicgan_2026/"                 # all outputs go under here
DATADIR="/mnt/processed/wolffjoa/restart_of_prediction_project/originalData"
BWDIR="${DATADIR}/gm12878/bigwig_CTCF/"        # chromatin features (CTCF.bigwig)

TRAIN_CHROMS="chr1 chr3 chr5 chr7 chr9 chr11 chr13 chr15 chr17 chr21"
VAL_CHROMS="chr19"
PRED_CHROMS="chr2 chr4 chr6 chr8 chr10 chr12 chr14 chr16 chr18 chr20"                             # NB: chr1 is in the training set
EPOCHS=100
BATCH=50
RECORDSIZE=2000
MULT=20000                                     # -mul, cooler scaling
TRAIN_GPU=2
PRED_GPU=1
MP="-mp"                                        # mixed precision (training only); set MP="" to disable
SAVEMEM="-sm"                                   # memory-bounded matrix rebuild (recommended at fine resolutions)
SKIP_EXISTING_TRAINING=1                        # 1 = skip training if a generator already exists

# training matrices per source resolution
declare -A MATRIX
MATRIX[5kb]="${DATADIR}/GSE63525_GM12878_insitu_primary+replicate_combined_30_5kb.cool"
MATRIX[10kb]="${DATADIR}/GSE63525_GM12878_insitu_primary+replicate_combined_30_10kb.cool"
MATRIX[25kb]="${DATADIR}/GM12878_25kb_cooler_coarsen.cool"

# training configs as "<sourceRes>:<windowSize>"
#   5kb and 10kb are each trained at BOTH window sizes 512 and 768; 25kb at 256
CONFIGS="25kb:256 10kb:512 5kb:512"

TARGETS_BP="1000 2000 5000 10000 25000 50000 100000"   # prediction resolutions (bp)

# ------------------------------------------------------------------ run -----
for cfg in ${CONFIGS}; do
    res="${cfg%%:*}"
    ws="${cfg##*:}"
    mat="${MATRIX[$res]}"
    traindir="${BASE}/train_${res}_ws${ws}"
    mkdir -p "${traindir}"

    if [[ "${SKIP_EXISTING_TRAINING}" == "1" ]] && ls "${traindir}"/generator_*.keras >/dev/null 2>&1; then
        echo "### [${res}] generator already present in ${traindir} -- skipping training"
    else
        echo "### [${res}] TRAINING  windowSize=${ws}  epochs=${EPOCHS}  -> ${traindir}"
        hicTraining \
            --trainingMatrices "${mat}" \
            --trainingChromosomes ${TRAIN_CHROMS} \
            --trainingChromatinFolders "${BWDIR}" \
            --validationMatrices "${mat}" \
            --validationChromosomes ${VAL_CHROMS} \
            --validationChromatinFolders "${BWDIR}" \
            --outputFolder "${traindir}" \
            --windowSize "${ws}" \
            --epochs "${EPOCHS}" \
            --batchSize "${BATCH}" \
            --recordSize "${RECORDSIZE}" \
            --lossWeightPixel 100 \
            --lossWeightDiscriminator 0.5 \
            --lossTypePixel L2 \
            --lossWeightTV 1e-12 \
            --lossWeightAdversarial 1.0 \
            --learningRateGenerator 2e-5 \
            --learningRateDiscriminator 1e-6 \
            --beta1 0.5 \
            --whichGPU "${TRAIN_GPU}" \
            ${MP}
    fi

    # pick the latest generator checkpoint (natural sort -> highest epoch)
    model="$(ls -1v "${traindir}"/generator_*.keras | tail -1)"
    echo "### [${res}] using model: ${model}"

    for bp in ${TARGETS_BP}; do
        kb="$((bp / 1000))kb"
        preddir="${BASE}/pred_${res}_ws${ws}/to_${kb}"
        mkdir -p "${preddir}"
        echo "### [${res}] PREDICT -> ${kb}  (binSize=${bp}, windowSize=${ws})"
        hicPredict \
            --trainedModel "${model}" \
            -tcp "${BWDIR}" \
            -pc ${PRED_CHROMS} \
            -mn "pred_${res}_to_${kb}.cool" \
            -o "${preddir}/" \
            --binSize "${bp}" \
            -mul "${MULT}" \
            -ws "${ws}" \
            --whichGPU "${PRED_GPU}" \
            --mode all \
            ${SAVEMEM}
    done
done

echo "### ALL DONE. Outputs under ${BASE}/ (train_*/  and pred_*/to_*/)"
