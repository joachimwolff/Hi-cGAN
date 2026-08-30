# Hyperparameter optimization scoring functions
The source code for the hyperparameter scoring function publication is provided in a separate repository for better differentiation from Hi-cGAN: https://github.com/joachimwolff/hyperparameterScoringHiC

# Hi-cGAN

Hi-cGAN is a conditional generative adversarial network 
designed to predict Hi-C contact matrices from one-dimensional
chromatin feature data, e. g. from ChIP-seq experiments.
The network architecture is inspired by [pix2pix from Isola et al.](https://doi.org/10.1109/CVPR.2017.632), amended by custom embedding networks to embed the one-dimensional chromatin feature data into grayscale images. 

Hi-cGAN was created in 2020/2021 as part of a master thesis at Albert-Ludwigs university, Freiburg, Germany. It is provided under the [GPLv3 license](https://github.com/joachimwolff/Hi-cGAN/blob/main/LICENSE).

## Installation

Hi-cGAN has been designed for Linux operating systems (tested under Ubuntu 24.04 and Rocky Linux 9). Other operating systems are not supported and probably won't work.

Clone this repository and install it into an empty environment:

```
git clone https://github.com/joachimwolff/Hi-cGAN.git
cd Hi-cGAN
pip install .
```

The installation provides five commands: `hicTraining`, `hicPredict`, `hicComputeCorrelation`, `hicOptimizer` and `hicScoring`. All of them accept `--version`.

The stack the released results were produced with:

dependency | tested version
-----------|---------------
python | 3.11.9
tensorflow | 2.15.0
keras | 2.15.0
cooler | 0.10.3
h5py | 3.11.0
numpy | 1.26.4
pandas | 2.2.2
pyBigWig | 0.3.22
scipy | 1.14.0
scikit-learn | 1.6.1
matplotlib | 3.8.4
hicmatrix | 17.2
tqdm | 4.66.4

Other versions might work but are untested. A CUDA capable GPU is needed for training.

Models are saved in the Keras v3 format (`generator_00099.keras`). Loading one requires `safe_mode=False`, because the architecture contains a Lambda layer:

```
model = keras.models.load_model("generator_00099.keras", compile=False, safe_mode=False)
```

## Input data requirements
* Hi-C matrix / matrices in cooler format for training.   
Cooler files must be single resolution (e.g. 25kbp). Multi-resolution files (mcool) are not supported.
* Chromatin features in bigwig format for training.   
Chromatin features and Hi-C matrix for training should be from the same cell line
and must use the same reference genome. File extension must be 'bigwig', 'bigWig' or 'bw'.
* Chromatin features in bigwig format for test / prediction.
Chromatin features for prediction must be the same as for training,
but of course for the cell line to be predicted.
The basic file names must be the same as for training.
See example usage for details.

Any one-dimensional signal stored as bigwig can be used, and several tracks are combined by placing them in one folder.

## Usage

### Training

`hicTraining` trains generator and discriminator by alternately updating their weights with the Adam optimizer. The generator uses a combined loss (L1 or L2 pixel loss, adversarial loss, total variation loss) and the discriminator standard binary cross entropy.

Training samples are generated per chromosome with a sliding window over the Hi-C matrix and the chromatin features, as proposed by [Farré et al.](https://doi.org/10.1186/s12859-018-2286-z). The two parameters that decide what a model can see are the window size and the bin size of the matrix: a window of `w` at a bin size of `b` reaches `w * b` base pairs from the diagonal, and the model cannot predict the last `w` bins of a chromosome.

Synopsis: `hicTraining [parameters and options]`

Required:

parameter | short | meaning
----------|-------|--------
--trainingMatrices | -tm | Hi-C matrices for training, cooler format. Repeat the option for more than one matrix; the first matrix belongs to the first training chromatin folder and so on
--trainingChromosomes | -tchroms | chromosomes for training, without leading "chr" and separated by spaces, e.g. "1 3 5 11". Must be present in all training matrices
--trainingChromatinFolders | -tcp | folder holding the bigwig files for training, one folder per training matrix, in the same order. Subfolders are not considered
--validationMatrices | -vm | matrices for validation, same rules as --trainingMatrices
--validationChromosomes | -vchroms | chromosomes for validation. Should not intersect the training chromosomes
--validationChromatinFolders | -vcp | chromatin folders for validation
--windowSize | -ws | window size in bins, 64, 128 or 256
--outputFolder | -o | output folder, must be writable
--epochs | -ep | number of epochs

Optional, the network and its losses:

parameter | short | default | meaning
----------|-------|---------|--------
--batchSize | -bs | 32 | batch size. With `--multiGPUTraining` this is the GLOBAL batch and is split across replicas
--lossWeightPixel | -lwp | 100.0 | weight of the L1 or L2 loss in the generator
--lossWeightDiscriminator | -lwd | 0.5 | weight of the discriminator error
--lossTypePixel | -ltp | L1 | per pixel loss, "L1" or "L2"
--lossWeightTV | -lwt | 1e-10 | weight of the total variation loss, higher means smoother
--lossWeightAdversarial | -lwa | 1.0 | weight of the adversarial loss in the generator
--learningRateGenerator | -lrg | 2e-5 | learning rate of the generator's Adam optimizer
--learningRateDiscriminator | -lrd | 1e-6 | learning rate of the discriminator's Adam optimizer
--beta1 | -b1 | 0.5 | beta1 of both Adam optimizers
--flipSamples | -fs | off | flip matrices and features as data augmentation

Optional, what enters training:

parameter | short | default | meaning
----------|-------|---------|--------
--minTargetCoverage | -mtc | 0.0 | drop every sample whose target window has a smaller fraction of covered bins than this. Without it a gap in the matrix becomes a block of zeros the model learns to reproduce
--excludeRegions | -exr | none | BED file or files; every sample whose target window overlaps one of these regions is dropped. Use it to hold regions out deliberately, for instance another method's test set
--noScaleNorm | -nsn | off | do not scale the chromatin features to the 0 to 1 range
--scaleTargetToUnitRange | -stur | off | map the targets onto [0, 1] using the range measured from the training targets, and write that range to `target_value_range.json` so `hicPredict` can invert it
--targetValueRange | -tvr | none | apply the mapping at read time for records written without it. Takes the path to `target_value_range.json`, `"min,max"`, or `"auto"`
--seed | -sd | none | seed for the Python, NumPy and TensorFlow generators. Set it to repeat a run, vary it for independent runs of one configuration

Optional, how it runs:

parameter | short | default | meaning
----------|-------|---------|--------
--multiGPUTraining | -mgpu | off | train on every visible GPU with MirroredStrategy
--whichGPU | -wgpu | 1 | which GPU to use in the single GPU case, one based
--mixedPrecision | -mp | off | float16 training, faster and smaller on the GPU, no effect on system RAM
--createDataOnly | -cdo | off | write the TFRecords and exit. Needs no GPU
--trainOnly | -to | off | train from existing TFRecords in the output folder
--threads | -t | 4 | threads for writing TFRecords
--recordSize | -rs | 2000 | approximate number of samples per TFRecord file
--keepTFRecords | -k | off | do not delete the TFRecords after training
--resume | -r | off | resume from the newest checkpoint in the output folder
--saveMemory | -sm | off | do not hold all data in memory at once
--validationSteps | -vs | 0 | evaluate only this many validation batches per epoch, 0 means all
--plotFrequency | -pfreq | 10 | update the loss plots every this many epochs
--figureFileFormat | -ft | png | png, pdf or svg

Returns, in the folder given by `-o`:
* generator and discriminator in Keras v3 format, written every `-pfreq` epochs and after the last epoch
* sample images of generated Hi-C matrices
* a parameter file in csv format
* the TFRecords, deleted after training unless `--keepTFRecords` is given

Because `--createDataOnly` needs no GPU and `--trainOnly` reads what it wrote, a run can be split into a CPU job that prepares the data and a GPU job that trains on it, both pointed at one output folder.

### Predict

`hicPredict` predicts a Hi-C matrix from chromatin features and a trained generator.

Synopsis: `hicPredict [parameters and options]`

parameter | short | default | meaning
----------|-------|---------|--------
--trainedModel | -trm | | the generator written by `hicTraining`
--predictionChromosomesFolders | -tcp | | folder with the bigwig files to predict from. Number and base names must match training
--predictionChromosomes | -pc | | chromosomes to predict, without "chr" and separated by spaces
--binSize | -b | | bin size for binning the chromatin features. This is the resolution of the prediction
--windowSize | -ws | | window size, must be the one the model was trained with
--outputFolder | -o | ./ | output folder
--matrixOutputName | -mn | predMatrix.cool | name of the predicted cooler
--parameterOutputFile | -pf | predParams.csv | name of the parameter file
--batchSize | -bs | 32 | batch size
--multiplier | -mul | 10 | scaling applied to the predicted cooler
--targetValueRange | -tvr | none | undo the [0, 1] mapping so the prediction comes back in the target's units. Path to `target_value_range.json` or `"min,max"`
--includeRegions | -ir | none | BED file or files; predict only the window positions whose target lies entirely inside these regions
--mode | -m | all | `create-data` (CPU only), `predict` (needs a GPU), `make-matrix`, or `all`
--rebuildProcesses | -rp | 0 | how many chromosomes to rebuild at once, 0 picks a number from free memory and core count
--saveMemory | -sm | off | memory saving mode
--numberOfBatches | -nb | 20 | batches to split the prediction into when `--saveMemory` is set
--whichGPU | -wgpu | 1 | which GPU to use, one based

Returns the predicted matrix in cooler format for the requested chromosomes, and a parameter file. The values are scaled for display by `--multiplier`; with `--targetValueRange` they are written back in the value range of the training target, so the result is an ordinary cooler.

### Example usage

Hi-C and chromatin features are available for cell_line1, and the same features for cell_line2. Hi-cGAN is trained on cell_line1 to predict the unknown matrix of cell_line2.

```
#following folder structure is assumed
#./cell_line1/feature1.bigwig, feature2.bigwig, feature3.bigwig, HiCmatrix_25kb.cool
#./cell_line2/feature1.bigwig, feature2.bigwig, feature3.bigwig
#./trained_models/
#./predictions/

tm="./cell_line1/HiCmatrix_25kb.cool"   #training matrix, 25 kbp bins
tcp="./cell_line1/"                     #folder with the training features
tchroms="1 3 5 7 9 11 13 15 17 21"      #training chromosomes
vm="./cell_line1/HiCmatrix_25kb.cool"   #validation matrix, here the same file
vcp="./cell_line1/"
vchroms="19"                            #must not intersect the training chromosomes

#train for 100 epochs with a window of 256, seeded so the run can be repeated
hicTraining -tm ${tm} -tcp ${tcp} -tchroms ${tchroms} \
            -vm ${vm} -vcp ${vcp} -vchroms ${vchroms} \
            -ws 256 -ep 100 -bs 2 --seed 42 -o ./trained_models

#predict cell_line2 with the trained generator
hicPredict -trm ./trained_models/generator_00099.keras \
           -tcp ./cell_line2/ -pc "2 4 6 8" \
           -b 25000 -ws 256 -o ./predictions

#the result is ./predictions/predMatrix.cool
hicPlotMatrix -m ./predictions/predMatrix.cool --region 2:0-1000000 --log1p -o cell_line2_chr2.png
```

Splitting the same run over a CPU and a GPU job, both writing to one output folder:

```
hicTraining ... -o ./trained_models --createDataOnly --threads 16
hicTraining ... -o ./trained_models --trainOnly --mixedPrecision
```

### Other commands

`hicComputeCorrelation` compares a predicted matrix against a measured one, `hicOptimizer` searches hyperparameters, and `hicScoring` applies the scoring functions. Each is documented by its own `--help`.

## Citation

Hi-cGAN v1 is the version described in the article. The models, predictions, processed inputs and analysis code are deposited at [10.5281/zenodo.11402891](https://doi.org/10.5281/zenodo.11402891).

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
