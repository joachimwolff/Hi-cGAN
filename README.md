# Hyperparameter optimization scoring functions
The source code for the hyperparameter scoring function publication is provided in a separate repository for better differentiation from Hi-cGAN: https://github.com/joachimwolff/hyperparameterScoringHiC

# Hi-cGAN

Hi-cGAN is a conditional generative adversarial network 
designed to predict Hi-C contact matrices from one-dimensional
chromatin feature data, e. g. from ChIP-seq experiments.
The network architecture is inspired by [pix2pix from Isola et al.](https://doi.org/10.1109/CVPR.2017.632), amended by custom embedding networks to embed the one-dimensional chromatin feature data into grayscale images. 

Hi-cGAN was created in 2020/2021 as part of a master thesis at Albert-Ludwigs university, Freiburg, Germany. It is provided under the [GPLv3 license](https://github.com/joachimwolff/Hi-cGAN/blob/main/LICENSE).

## Installation

Tested under Ubuntu 24.04 with Python 3.11.9, TensorFlow 2.15.0, cooler 0.10.3, numpy 1.26.4, pandas 2.2.2, pyBigWig 0.3.22 and h5py 3.11.0. Other versions might work but are untested. A CUDA capable GPU is needed for training.

```
git clone https://github.com/joachimwolff/Hi-cGAN.git
cd Hi-cGAN
pip install .
```

This installs `hicTraining`, `hicPredict`, `hicComputeCorrelation`, `hicOptimizer` and `hicScoring`. Models are written in the Keras v3 format and load with `load_model(path, compile=False, safe_mode=False)`, the flag being needed because the architecture contains a Lambda layer.

## Input data requirements
* Hi-C matrix in cooler format, single resolution. Multi-resolution files (mcool) are not supported.
* Chromatin features as bigwig, in one folder per matrix, file extension 'bigwig', 'bigWig' or 'bw'. Any one-dimensional signal stored as bigwig works, and several tracks are combined by putting them in one folder.
* Features and matrix must come from the same cell line and reference genome.
* For prediction, the same tracks under the same base names, for the cell line to be predicted.

## Performance

GM12878, trained on the odd chromosomes with chromosome 19 for validation, scored at epoch 100 on the twelve held-out chromosomes against the measured matrices of [Rao et al.](https://doi.org/10.1016/j.cell.2014.11.021):

bin size | input tracks | window | HiCRep SCC | GenomeDISCO | HiC-Spector | insulation r
---------|--------------|--------|------------|-------------|-------------|-------------
25 kb | H3K27ac, CTCF | 256 | 0.586 | 0.735 | 0.568 | 0.749
10 kb | CTCF, H3K4me2 | 512 | 0.623 | 0.709 | 0.598 | 0.785
5 kb | RAD21, H3K4me3, SMC3 | 512 | 0.643 | 0.665 | 0.478 | 0.750

The most informative track depends on the resolution: CTCF and the cohesin subunits at 5 to 10 kb, active histone marks at 25 kb. Two tracks are usually enough. Predicting a cell type the model never saw costs about 0.12 SCC. Boundaries and loops called from the maps agree less well than these numbers suggest, so the maps suit domain-scale description rather than boundary or loop calling.

Training 100 epochs on one Nvidia A100 takes 1.7 h at 25 kb with a window of 64 and 149 h at 5 kb with a window of 512. Predicting a whole genome on one RTX 4090 takes 1.4 min at 25 kb and 28 min at 5 kb, with a peak of 4.5 GB on the card.

## Trained models and data

Models, predictions, processed inputs and analysis code are deposited at [10.5281/zenodo.11402891](https://doi.org/10.5281/zenodo.11402891). `02_best_models.tar.gz` holds the generator and the whole-genome prediction for each bin size, the other archives the factor searches per resolution, the 25 kb epoch scan and the code with the harvested scores.

Checkpoints are named `generator_000NN.keras` with `NN` the zero-based epoch, so `generator_00099.keras` is the state after the 100th epoch, the budget every number above was read at. Only generators are deposited; the discriminator is needed for training, not for prediction.

bin size | window | input tracks
---------|--------|-------------
25 kb | 256 | H3K27ac, CTCF
10 kb | 512 | CTCF, H3K4me2
5 kb | 512 | RAD21, H3K4me3, SMC3

All three were trained on the odd chromosomes of GM12878. The 2 kb model belongs to the Akita comparison, which holds out windows rather than chromosomes, and is deposited with it.

## Usage

### Training

`hicTraining` trains generator and discriminator by alternately updating their weights with the Adam optimizer. Samples are cut per chromosome with a sliding window over matrix and features, as proposed by [Farré et al.](https://doi.org/10.1186/s12859-018-2286-z). A window of `w` at bin size `b` reaches `w * b` base pairs from the diagonal, and the last `w` bins of a chromosome cannot be predicted.

Required:

parameter | short | meaning
----------|-------|--------
--trainingMatrices | -tm | training matrices in cooler format, repeat for more than one. The first matrix belongs to the first chromatin folder
--trainingChromosomes | -tchroms | chromosomes for training, without "chr", separated by spaces
--trainingChromatinFolders | -tcp | folder with the bigwig files, one per training matrix, in the same order
--validationMatrices | -vm | as --trainingMatrices, for validation
--validationChromosomes | -vchroms | validation chromosomes, should not intersect the training ones
--validationChromatinFolders | -vcp | as --trainingChromatinFolders, for validation
--windowSize | -ws | window size in bins, 64, 128 or 256
--outputFolder | -o | output folder
--epochs | -ep | number of epochs

Worth knowing about, `hicTraining --help` lists the rest, including the loss weights and learning rates:

parameter | short | default | meaning
----------|-------|---------|--------
--batchSize | -bs | 32 | with --multiGPUTraining this is the global batch and is split across replicas
--minTargetCoverage | -mtc | 0.0 | drop samples whose target window has fewer covered bins than this fraction. Without it a gap in the matrix becomes a block of zeros the model learns to reproduce
--excludeRegions | -exr | none | BED files whose regions are kept out of training, for instance another method's test set
--seed | -sd | none | seed for Python, NumPy and TensorFlow, to repeat a run or to vary independent runs
--mixedPrecision | -mp | off | float16 training, faster and smaller on the GPU
--multiGPUTraining | -mgpu | off | train on every visible GPU
--whichGPU | -wgpu | 1 | which GPU in the single GPU case, one based
--createDataOnly | -cdo | off | write the TFRecords and exit, needs no GPU
--trainOnly | -to | off | train from existing TFRecords
--resume | -r | off | resume from the newest checkpoint in the output folder

Because `--createDataOnly` needs no GPU and `--trainOnly` reads what it wrote, a run splits into a CPU job and a GPU job sharing one output folder.

### Predict

`hicPredict` predicts a matrix from chromatin features and a trained generator. The bin size and window are properties of the checkpoint and must be passed unchanged.

parameter | short | default | meaning
----------|-------|---------|--------
--trainedModel | -trm | | the generator written by hicTraining
--predictionChromosomesFolders | -tcp | | folder with the bigwig files, same base names as in training
--predictionChromosomes | -pc | | chromosomes to predict, without "chr", separated by spaces
--binSize | -b | | bin size, this is the resolution of the prediction
--windowSize | -ws | | window size the model was trained with
--outputFolder | -o | ./ | output folder
--targetValueRange | -tvr | none | undo the [0, 1] mapping so the result comes back in the units of the training target
--includeRegions | -ir | none | predict only the window positions lying inside these BED regions
--mode | -m | all | create-data (CPU), predict (GPU), make-matrix, or all

Returns the predicted matrix in cooler format and a parameter file. `hicPredict --help` lists the remaining options.

### Example usage

```
#./cell_line1/ holds feature1.bigwig, feature2.bigwig and HiCmatrix_25kb.cool
#./cell_line2/ holds the same features for the cell line to be predicted

hicTraining -tm ./cell_line1/HiCmatrix_25kb.cool -tcp ./cell_line1/ \
            -tchroms "1 3 5 7 9 11 13 15 17 21" \
            -vm ./cell_line1/HiCmatrix_25kb.cool -vcp ./cell_line1/ -vchroms "19" \
            -ws 256 -ep 100 -bs 2 --seed 42 -o ./trained_models

hicPredict -trm ./trained_models/generator_00099.keras -tcp ./cell_line2/ \
           -pc "2 4 6 8" -b 25000 -ws 256 -o ./predictions

hicPlotMatrix -m ./predictions/predMatrix.cool --region 2:0-1000000 --log1p -o chr2.png
```

`hicComputeCorrelation` compares a prediction against a measured matrix, `hicOptimizer` searches hyperparameters and `hicScoring` applies the scoring functions.

## Notes
### Creating bigwig files for chromatin features from BAM alignment files
If bigwig files of the chromatin features are not available,
it is possible to use `bamCoverage` [[link]](https://github.com/deeptools/deepTools/blob/master/docs/content/tools/bamCoverage.rst) to convert alignments in .bam format to bigwig
for example as shown below.
```
# creating a bigwig file from the bam file BAMFILE (which ends in ".bam")
OUTFILE="${BAMFILE%bam}bigwig"
hg19SIZE="2685511504" #e.g. human ref. genome hg19. Adjust as needed.
COMMAND="--numberOfProcessors 10 --bam ${BAMFILE}"
COMMAND="${COMMAND} --outFileName $ {OUTFILE}"
COMMAND="${COMMAND} --outFileFormat bigwig"
COMMAND="${COMMAND} --binSize 5000 --normalizeUsing RPGC"
COMMAND="${COMMAND} --effectiveGenomeSize $ {hg19SIZE}"
COMMAND="${COMMAND} --scaleFactor 1.0 --extendReads 200"
COMMAND="${COMMAND} --minMappingQuality 30"
bamCoverage ${COMMAND}
```

If data for more than one replicate is available,
it is possible to merge replicates by first converting to bigwig as shown above  and then taking the mean across replicates using `bigwigCompare` from deeptools suite [[link]](https://github.com/deeptools/deepTools) for example like so:
```
#REPLICATE1 and REPLICATE2 are bigwig files
COMMAND="-b1 ${REPLICATE1} -b2 ${REPLICATE2}"
COMMAND="${COMMAND} -o ${OUTFILE} -of bigwig"
COMMAND="${COMMAND} --operation mean -bs 5000"
COMMAND="${COMMAND} -p 10 -v"
bigwigCompare ${COMMAND}
```

### Creating bigwig files for chromatin features from fastq files
If no alignments in bam format are available, most published ChIP-seq experiments (or similar) at least offer fastq or fastqsanger files for download from Sequence Read Archive. Download these and map them with a mapping tool suitable for the type of experiment, e.g. bowtie, bowtie2, bwa-mem. The parameters for these tools depend on the type of experiment and possibly some preprocessing done by the authors, so no recommendations can be made here. With respect to Hi-cGAN, the only requirement is to use the same reference genome as for the Hi-C matrix.
After computing alignments, convert to BAM format, if necessary, and proceed as shown above for BAM files.


### Creating cooler files
Cooler offers a bunch of tools for converting Hi-C matrices from other formats
into cooler format, e.g. `hic2cool`. Check https://github.com/open2c/cooler
