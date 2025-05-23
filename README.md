# Hyperparameter optimization scoring functions
The source code for the hyperparameter scoring function publication is provided in a separate repository for better differentiation from Hi-cGAN: https://github.com/joachimwolff/hyperparameterScoringHiC

# Hi-cGAN

Hi-cGAN is a conditional generative adversarial network 
designed to predict Hi-C contact matrices from one-dimensional
chromatin feature data, e. g. from ChIP-seq experiments.
The network architecture is inspired by [pix2pix from Isola et al.](https://doi.org/10.1109/CVPR.2017.632), amended by custom embedding networks to embed the one-dimensional chromatin feature data into grayscale images. 

Hi-cGAN was created in 2020/2021 as part of a master thesis at Albert-Ludwigs university, Freiburg, Germany. It is provided under the [GPLv3 license](https://github.com/MasterprojectRK/Hi-cGAN/blob/main/LICENSE).

## Installation

Hi-cGAN has been designed for Linux operating systems (tested under Ubuntu 20.04 and CentOS 7.9.2009). Other operating systems are not supported and probably won't work.

Simply `git clone` this repository into an empty folder of your choice.
It is recommended to use conda or another package manager to install
the following dependencies into an empty environment:
dependency | tested version
-----------|---------------
click | 7.1.2
cooler | 0.8.10
graphviz | 2.42.3
matplotlib | 3.3.2
numpy | 1.19.4
pandas | 1.1.4
pybigwig | 0.3.17
pydot | 1.4.1
python | 3.7.8
scikit-learn | 0.23.2
scipy | 1.5.3
tensorflow-gpu | 2.2.0   
tqdm | 4.50.2

Other versions *might* work, but are untested and might cause dependency
conflicts. Updating to tensorflow 2.3.x should be possible but has not been tested. Using tensorflow without GPU support is possible, but will be very slow and is thus not recommended. 


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


## Usage
Hi-cGAN consists of two python scripts, training.py and predict.py,
which will be explained below.

### Training
This script will train the cGAN Generator and Discriminator
by alternately updating their weights using the Adam optimizer.
Here, the generator features a combined loss function (L1/L2 loss, adversarial loss, TV Loss) and the discriminator is using standard binary cross entropy loss.  

Hi-cGAN is using a sliding window approach to generate training samples (and test samples, too) from Hi-C matrices and chromatin features on a per-chromosome basis, as proposed by [Farré et al.](https://doi.org/10.1186/s12859-018-2286-z). The most important parameters here are the window size (64, 128 or 256) and the bin size of the Hi-C matrix (e.g. 5kbp, 10kbp, 25kbp).

Synopsis: `python training.py [parameters and options]`  
Parameters / Options:  
- --trainmatrices | -tm 
  - required  
  - Hi-C matrices for training 
  - must be in cooler format 
  - use this option multiple times to specify more than one matrix (e.g. `-tm matrix1.cool -tm matrix2.cool`)
  - first matrix belongs to first training chromatin feature path and so on, see below
- --trainChroms | -tchroms 
  - required  
  - chromosomes for training
  - specify without leading "chr" and separated by spaces,
e.g. "1 3 5 11" 
  - these chromosomes must be present in all train matrices
- --trainChromPaths | -tcp 
  - required
  - path where chromatin features for training reside
  - program will look for bigwig files in this folder, subfolders are not considered
  - file extension must be "bigwig", "bigWig" or "bw"
  - specify one trainChromPath for each training matrix, in the desired order
  - chromatin features for training and prediction must have the same base names 
- --valMatrices | -vm 
  - required  
  - Hi-C matrices for validation
  - must be in cooler format. 
  - use this option multiple times to specify more than one matrix
- --valChroms | -vchroms 
  - required  
  - same as trainChroms, just for validation
- --valChromPaths | -vcp 
  - required
  - same as trainChromPaths, just for validation
- --windowsize | -ws  
  - required
  - window size in bins for submatrices in sliding window approach 
  - choose from 64, 128, 256 
  - default: 64
  - choose reasonable value according to matrix bin size
  - if the matrix has a bin size of 5kbp, then a windowsize of 64 corresponds to an actual windowsize of 64*5kbp = 320kbp
- --outfolder | -o 
  - required
  - folder where output will be stored
  - must be writable and have several 100s of MB of free storage space
- --epochs | -ep 
  - required
  - number of epochs for training 
- --batchsize | -bs 
  - required  
  - batch size for training
  - integer between 1 and 256
  - default: 32 
  - mind the memory limits of your GPU
  - in a test environment with 15GB GPU memory, batchsizes 32,4,2 were safely within limits for windowsizes 64,128,256, respectively
- --lossWeightPixel | -lwp 
  - optional 
  - loss weight for the L1 or L2 loss in the generator
  - float >= 1e-10
  - default: 100.0 
- --lossWeightDisc | -lwd  
  - optional
  - loss weight for the discriminator error
  - float >= 1e-10
  - default: 0.5
- --lossTypePixel | -ltp 
  - optional 
  - type of per-pixel loss to use for the generator
  - choose from "L1" (mean abs. error) or "L2" (mean squared error)
  - default: L1
- --lossWeightTv | -lwt 
  - optional 
  - loss weight for Total-Variation-loss of generator
  - float >= 0.0
  - default: 1e-10
  - higher value - more smoothing
- --lossWeightAdv | -lwa   
  - optional
  - loss weight for adversarial loss in the generator
  - float >= 1e-10
  - default: 1.0
- --learningRateGen | -lrg  
  - optional
  - learning rate for the Adam optimizer of the generator
  - float in 1e-10...1.0
  - default: 2e-5
- --learningRateDisc | -lrd
  - optional
  - learning rate for the Adam optimizer of the discriminator
  - float in 1e-10...1.0
  - default: 1e-6
- --beta1 | -b1
  - optional 
  - beta1 parameter for the Adam optimizers (generator and discriminator)
  - float in 1e-2...1.0  
  - default 0.5.
- --flipsamples | -fs 
  - optional
  - flip training matrices and chromatin features (data augmentation)
  - boolean
  - default: False
- --embeddingType | -emb 
  - optional  
  - type of embedding to use for generator and discriminator
  - choose from 'CNN' (convolutional neural network), 'DNN' (dense neural network by [Farré et al.](https://doi.org/10.1186/s12859-018-2286-z)), or 'mixed' (Generator - CNN, Discriminator - DNN)
  - default: CNN
  - CNN is recommended
- --pretrainedIntroModel | -ptm
  - optional  
  - undocumented, developer use only
- --figuretype | -ft  
  - optional
  - figure type for all plots
  - choose from png, pdf, svg 
  - default: png
- --recordsize | -rs
  - optional
  - approx. size (number of samples) of the tfRecords used in the data pipeline for training
  - can be tweaked to balance the load between RAM / GPU / CPU
  - integer >= 10
  - default: 2000
- --plotFrequency | -pfreq
  - optional
  - update and save loss over epoch plots after this number of epochs 
  - integer >= 1
  - default: 10

Returns: 
* The following files will be stored in the chosen output path (option `-o`) 
* Trained models of generator and discriminator in h5py format, stored in output path (every `-pfreq` epochs and after completion).
* Sample images of generated Hi-C matrices (every 5 epochs).
* Parameter file in csv format for reference.
* (temporary) Tensorflow TFRecord files containing serialized train samples. Do not touch these files while the program is running, they should be open for reading anyway and will be deleted automatically upon completion.


### Predict
This script will predict Hi-C matrices using chromatin features and a trained generator model as input.  

Synopsis: `python predict.py [parameters and options]`  
Parameters / Options:  
- --trainedModel | -trm 
  - required
  - trained generator model to predict from, h5py format
  - generated by training.py above
- --testChromPath | -tcp 
  - required
  - Same as trainChromPaths, just for testing / prediction
  - number and base names of bigwig files in this path must be the same as for training
- --testChroms | -tchroms
  - required
  - chromosomes for testing (to be predicted) 
  - must be available in all bigwig files
  - input format: without "chr" and separated by spaces, e.g. "8 12 21"
- --outfolder | -o
  - required
  - output path for predicted Hi-C matrices (in cooler format)
  - default: current path
- --multiplier | -mul 
  - optional
  - multiplier for better visualization of results
  - integer >= 1
  - default: 1000 
- --binsize | -b 
  - required
  - bin size for binning the proteins
  - usually equal to binsize for training (but not mandatory)
  - integer >= 1000
* --batchsize | -bs
  - optional
  - batch size for prediction
  - same considerations as for training.py hold
  - integer >= 1
  - default: 32
- --windowsize | -ws  
  - required
  - window size for prediction
  - choose from 64, 128, 256
  - must be the same as for training
  - could in future be detected from trained model
  - for now, just enter the appropriate value

Returns:  
* Predicted matrix in cooler format, defined for the specified test chromosomes.  
* Parameter file in csv format for reference.  
* (temporary) Tensorflow TFRecord files containing serialized prediction samples. Do not touch these files while the program is running, they should be open for reading anyway and will be deleted automatically upon completion.

### Example usage
Assume Hi-C and chromatin feature data is available for cell_line1,
and the same chromatin feature is also available for cell_line2.
Then Hi-cGAN can be trained on data from cell_line1 to predict
cell_line2's (unknown) Hi-C matrix.
```
#following folder structure is assumed
#./
#./cell_line1/
#./cell_line1/feature1.bigwig
#./cell_line1/feature2.bigwig
#./cell_line1/feature3.bigwig
#./cell_line1/HiCmatrix.cool
#./cell_line2/
#./cell_line2/feature1.bigwig
#./cell_line2/feature2.bigwig
#./cell_line2/feature3.bigwig
#./trained_models/
#./predictions/

#training Hi-C matrix
#assuming it has a 25kbp bin size
tm="./cell_line1/HiCmatrix_25kb.cool"
#training chromatin features
tcp="./cell_line1/"
#training chromosomes
tchroms="1 5 10"
#validation matrix, chromatin features and chromosome(s). 
vm="./cell_line1/HiCmatrix.cool" #here, same as for training
vcp="./cell_line1/" #here, same as for training
vchroms="19" #here, should not intersect with training chromosomes

#train Hi-cGAN on data from cell_line1, 100 epochs 
#this might take several hours to days, depending on hardware
#GPU strongly recommended for windowsizes 128, 256
#progress bars are provided for runtime estimation
training.py -tm ${tm} -tcp ${tcp} -tchroms ${tchroms} -vm ${vm} -vcp ${vcp} -vchroms ${vchroms} -o ./trained_models -ep 100

#the trained model with weights etc.
#this file is generated by running training.py as shown above
trm="./trained_models/generator_00099.h5"
#the chromatin path for prediction
tcp="./cell_line2/"
#the chromosomes to be predicted
tchroms="3 7 21"
#the binsize of the target matrix
#here, same as for training
b="25000"
#the windowsize of the target matrix
#must be the same as for training
ws="64"


#now use the trained model from above to predict Hi-C matrix for cell line 2
predict.py -trm ${trm} -tcp ${tcp} -tchroms ${tchroms} -o ./predictions -b ${b} -ws ${ws}

#the prediction script often completes within a few minutes on recent hardware
#after that, there's a file named ./predictions/predMatrix.cool
#which holds the predicted Hi-C matrix (here, with chromosomes 3, 7 and 21)
#e.g. plot the matrix 
hicPlotMatrix -m ./predictions/predMatrix.cool --region 3:0-1000000 --log1p -o cell_line2_chr3_0000000-1000000.png
```

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
