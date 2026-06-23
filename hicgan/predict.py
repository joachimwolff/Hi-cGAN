import argparse
import numpy as np
import os
import csv
import tensorflow as tf
from scipy import sparse
from .lib import dataContainer
from .lib import records
from .lib import hicGAN
from .lib import utils
import logging
from hicgan._version import __version__
import pickle

log = logging.getLogger(__name__)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--trainedModel", "-trm", required=False,
                        type=str,
                        help="Trained generator model to predict from")
    parser.add_argument("--predictionChromosomesFolders", "-tcp", required=False,
                        type=str,
                        help="Path where test data (bigwig files) resides")
    parser.add_argument("--predictionChromosomes", "-pc", required=False,
                        type=str,
                        nargs='+',
                        help="Chromosomes the Hi-C matrix should be predicted. Must be available in all bigwig files")
    parser.add_argument("--matrixOutputName", "-mn", required=False,
                        type=str,
                        default="predMatrix.cool",
                        help="Name of the output cool-file")
    parser.add_argument("--parameterOutputFile", "-pf", required=False,
                        type=str,
                        default="predParams.csv",
                        help="Name of the parameter file")
    parser.add_argument("--outputFolder", "-o", required=False,
                        type=str,
                        default="./", 
                        help="Output path for predicted cool-file")
    parser.add_argument("--multiplier", "-mul", required=False,
                        type=int, 
                        default=10, 
                        help="Multiplier for scaling the predicted coolers")
    parser.add_argument("--binSize", "-b", required=False,
                        type=int, 
                        help="Bin size for binning the chromatin features")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int,
                        default=32, 
                        help="Batch size for predicting")
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int,
                        choices=[64, 128, 256, 512, 768, 1024],
                        help="Window size for predicting; must be the same as in trained model. Supported values are 64, 128, and 256")
    parser.add_argument("--saveMemory", "-sm", action="store_true",
                        help="Enable memory-saving mode for prediction")
    parser.add_argument("--numberOfBatches", "-nb", required=False,
                        type=int,
                        default=20,
                        help="Number of batches to split predictions when --saveMemory is enabled")
    parser.add_argument("--whichGPU", "-wgpu", required=False,
                        type=int,
                        default="",
                        help="Specify which GPU to use for training in the single GPU case. E.g. 1, 2, etc.")
    parser.add_argument("--mode", "-m", required=False,
                        type=str,
                        choices=["create-data", "predict", "make-matrix", 'all'],
                        default="all",
                        help="Operation mode: 'create-data' (only create TFRecord data; CPU only), "
                                "'predict' (run prediction using the trained model; requires GPU), "
                                "'make-matrix' (build final matrix from existing predictions; CPU only), "
                                "'all' (run all steps sequentially)")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser

def createDataContainer(pChromNameList, pOutputFolder, pChromatinFolder, pBinSize, 
                        scalefactors, clampfactors, scalematrix, windowSize, flankingsize, maxdist, batchSize):
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in pChromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixFilePath=None,
                                                  chromatinFolder=pChromatinFolder,
                                                  binSize=pBinSize)) 
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowSize": windowSize,
                  "flankingSize": flankingsize,
                  "maximumDistance": maxdist}
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return
    # A sliding window spans windowSize + 2*flankingSize bins. At a coarse
    # binSize a chromosome may have fewer bins than that and would otherwise
    # crash deep inside getNumberSamples. Skip such chromosomes with a clear
    # warning instead of aborting the whole prediction run, so that
    # resolution-independent prediction works for whichever chromosomes fit.
    required_bins = windowSize + 2 * flankingsize
    container0 = None
    keptContainers = []
    tfRecordFilenames = []
    sampleSizeList = []
    for container in testdataContainerList:
        container.loadData(**loadParams)
        if not container.hasEnoughBins():
            nr_bins = container.FactorDataArray.shape[0] if container.FactorDataArray is not None else 0
            msg = ("Skipping chr{:s}: {:d} bins at binSize {:s} < required {:d} "
                   "(windowSize {:d} + 2x flanking). Use a finer binSize to include it.").format(
                       str(container.chromosome), nr_bins, str(pBinSize), required_bins, windowSize)
            print(msg)
            log.warning(msg)
            container.unloadData()
            continue
        if container0 is None:
            container0 = container
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
            raise SystemExit(msg)
        tfRecordFilenames.append(container.writeTFRecord(pOutputFolder=pOutputFolder,
                                                        pRecordSize=None)[0])
        sampleSizeList.append(int(np.ceil(container.getNumberSamples() / batchSize)))
        keptContainers.append(container)

    if container0 is None:
        msg = ("Exiting. No requested chromosome has enough bins at binSize {:s} for windowSize {:d} "
               "(need >= {:d} bins). Predict at a finer binSize (smaller -b).").format(
                   str(pBinSize), windowSize, required_bins)
        print(msg)
        raise SystemExit(msg)

    nr_factors = container0.nr_factors
    for container in keptContainers:
        container.unloadData()
    return keptContainers, tfRecordFilenames, sampleSizeList, nr_factors

def prediction(pTrainedModel, pPredictionChromosomesFolders, pPredictionChromosomes, pOutputFolder, pMultiplier, pBinSize, pBatchSize, pWindowSize, pMatrixOutputName, pParameterOutputFile, pSaveMemory=False, pNumberOfBatches=20, pScope=None, pMode="all"):
    trainedmodel = pTrainedModel
    predictionChromosomesFolders = pPredictionChromosomesFolders
    predictionChromosomes = pPredictionChromosomes
    outputFolder = pOutputFolder
    multiplier = pMultiplier
    binSize = pBinSize
    batchSize = pBatchSize
    windowSize = pWindowSize

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    scalefactors = True
    clampfactors = False
    scalematrix = True
    maxdist = None
    windowSize = int(windowSize)
    flankingsize = windowSize

    paramDict = locals().copy()
    predictionChunksDir = os.path.join(outputFolder, "prediction_chunks")
    matrixChunksDir = os.path.join(outputFolder, "matrix_chunks")
    if not os.path.exists(predictionChunksDir):
        os.makedirs(predictionChunksDir)
    if not os.path.exists(matrixChunksDir):
        os.makedirs(matrixChunksDir)
        
    chromNameList = sorted([x.lstrip("chr") for x in predictionChromosomes])

    if pMode in ["create-data", "all"]:
        testdataContainerList, tfRecordFilenames, sampleSizeList, nr_factors = createDataContainer(
            pChromNameList=chromNameList, 
            pOutputFolder=outputFolder, 
            pChromatinFolder=predictionChromosomesFolders, 
            pBinSize=binSize, 
            scalefactors=scalefactors, 
            clampfactors=clampfactors, 
            scalematrix=scalematrix, 
            windowSize=windowSize, 
            flankingsize=flankingsize, 
            maxdist=maxdist, 
            batchSize=batchSize)
        if pMode == "create-data":
            save_vars = {
                "testdataContainerList": testdataContainerList,
                "tfRecordFilenames": tfRecordFilenames,
                "sampleSizeList": sampleSizeList,
                "nr_factors": nr_factors,
            }

            pickle_path = os.path.join(outputFolder, "prediction_vars.pkl")
            with open(pickle_path, "wb") as fh:
                pickle.dump(save_vars, fh, protocol=pickle.HIGHEST_PROTOCOL)

            log.info("Wrote prediction variables to %s", pickle_path)
            return
    elif pMode == "predict":
        pickle_path = os.path.join(outputFolder, "prediction_vars.pkl")
        if not os.path.exists(pickle_path):
            log.error("Pickle file with prediction variables not found at %s. Please run with --mode create-data first.", pickle_path)
            return
        
        with open(pickle_path, "rb") as fh:
            loaded_vars = pickle.load(fh)

        testdataContainerList = loaded_vars["testdataContainerList"]
        tfRecordFilenames = loaded_vars["tfRecordFilenames"]
        sampleSizeList = loaded_vars["sampleSizeList"]
        nr_factors = loaded_vars["nr_factors"]

    if pMode in ["predict", "all"]:
        trained_GAN = hicGAN.HiCGAN(log_dir=outputFolder, number_factors=nr_factors, scope=pScope)
        trained_GAN.loadGenerator(trainedModelPath=trainedmodel)
        predictionFiles = []
        predictedChroms = []
        # Derive the chromosome from the container itself rather than the
        # originally requested chromNameList: some chromosomes may have been
        # skipped (too few bins at this binSize), so positional zipping with
        # the full request list would misalign names and predictions.
        for record, container, nr_samples in zip(tfRecordFilenames, testdataContainerList, sampleSizeList):
            chrom = container.chromosome
            storedFeaturesDict = container.storedFeatures
            testDs = tf.data.TFRecordDataset(record,
                                                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                compression_type=records.TFRECORD_COMPRESSION or "")
            testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            testDs = testDs.batch(batchSize, drop_remainder=False)
            testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)

            # Stream predictions one batch at a time. Each batch is reduced to
            # its upper-triangle as float16 immediately, so the full-chromosome
            # float32 stack is never held in memory and the dataset is decoded
            # exactly once (the previous save-memory path used dataset.skip(),
            # which re-decodes the dataset O(numberOfBatches) times).
            triu_r, triu_c = np.triu_indices(windowSize)
            log.debug("Streaming predictions and extracting triangles...")
            predBatches = []
            for batch_pred in trained_GAN.predict_stream(test_ds=testDs):
                # vectorized upper-triangle extraction: (batch, W, W) -> (batch, n_triu)
                predBatches.append(batch_pred[:, triu_r, triu_c].astype(np.float16))
            if predBatches:
                predArray = np.concatenate(predBatches, axis=0).astype(np.float16)
            else:
                log.warning("No predictions produced for chr%s", chrom)
                predArray = np.empty((0, triu_r.size), dtype=np.float16)
            del predBatches

            predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")
            np.save(predFile, predArray, allow_pickle=False)
            predictionFiles.append(predFile)
            predictedChroms.append(chrom)
            log.info("Wrote predictions for chr%s to %s", chrom, predFile)
            del predArray
        log.debug("Prediction on all chromosomes completed.")
        # keep the downstream chromosome list consistent with what was predicted
        chromNameList = predictedChroms

        log.info("Cleaning up temporary files...")
        for tfrecordfile in tfRecordFilenames:
            if os.path.exists(tfrecordfile):
                os.remove(tfrecordfile)
        if pMode == "predict":
            save_vars = {
                "predictionFiles": predictionFiles,
                "chromNameList": chromNameList,
                "windowSize": windowSize,
            }

            pickle_path = os.path.join(outputFolder, "predictions.pkl")
            with open(pickle_path, "wb") as fh:
                pickle.dump(save_vars, fh, protocol=pickle.HIGHEST_PROTOCOL)

            log.info("Wrote predictions to %s", pickle_path)
            return
    if pMode == "make-matrix":
        pickle_path = os.path.join(outputFolder, "predictions.pkl")
        if not os.path.exists(pickle_path):
            log.error("Pickle file with predictions not found at %s. Please run with --mode predict first.", pickle_path)
            return
        
        with open(pickle_path, "rb") as fh:
            loaded_vars = pickle.load(fh)

        predictionFiles = loaded_vars.get("predictionFiles")
        chromNameList = loaded_vars["chromNameList"]
        if "windowSize" in loaded_vars:
            windowSize = int(loaded_vars["windowSize"])

        # Backward compatibility: older pickles stored all predictions in-memory.
        if predictionFiles is None and "predList" in loaded_vars:
            predictionFiles = []
            for chrom, predArray in zip(chromNameList, loaded_vars["predList"]):
                predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")
                np.save(predFile, np.asarray(predArray, dtype=np.float16), allow_pickle=False)
                predictionFiles.append(predFile)
            log.info("Converted legacy in-memory predictions to disk-backed chunks in %s", predictionChunksDir)

    if pMode in ["make-matrix", "all"]:
        if pMode == "all":
            # In all-mode the prediction step produced the chunk file list.
            if "predictionFiles" not in locals():
                predictionFiles = []
                for chrom in chromNameList:
                    predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")
                    if os.path.exists(predFile):
                        predictionFiles.append(predFile)

        if not predictionFiles:
            log.error("No prediction chunk files found in %s", predictionChunksDir)
            return

        log.debug("Rebuilding full matrices from predicted triangles one chromosome at a time...")
        matrixFiles = []
        for chrom, predFile in zip(chromNameList, predictionFiles):
            log.info("Loading prediction chunk for chr%s from %s", chrom, predFile)
            predTriangles = np.load(predFile, mmap_mode="r")
            predMatrix = utils.rebuildMatrix(pArrayOfTriangles=predTriangles, pWindowSize=windowSize, pFlankingSize=windowSize, pSaveMemory=pSaveMemory)
            predMatrix = utils.scaleArray(predMatrix) * multiplier

            matrixFile = os.path.join(matrixChunksDir, f"matrix_chr{chrom}.npz")
            if sparse.isspmatrix(predMatrix):
                sparse.save_npz(matrixFile, predMatrix.tocsr())
            else:
                sparse.save_npz(matrixFile, sparse.csr_matrix(predMatrix))
            matrixFiles.append(matrixFile)
            log.info("Wrote scaled matrix for chr%s to %s", chrom, matrixFile)

            del predTriangles
            del predMatrix

        matrixname = os.path.join(outputFolder, pMatrixOutputName)
        log.info("Writing predicted matrix to disk on %s..." % matrixname)   

        utils.writeCooler(pMatrixList=matrixFiles, 
                      pBinSizeInt=binSize, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)

        parameterFile = os.path.join(outputFolder, pParameterOutputFile) 
        with open(parameterFile, "w") as csvfile:
            dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
            dictWriter.writeheader()
            dictWriter.writerow(paramDict)
        
       


def main(args=None):
    args = parse_arguments().parse_args(args)
    # print(args)
    # 1. Get all physical GPUs

    if args.mode == "all" or args.mode == "predict":
        physical_gpus = tf.config.list_physical_devices('GPU')

        if physical_gpus:
            try:
                # Calculate index (assuming args.whichGPU is 1-based data like 1, 2, 3...)
                index = args.whichGPU - 1
                if index < 0 or index >= len(physical_gpus):
                    raise ValueError(f"Invalid GPU index: {index}. Available: {len(physical_gpus)}")

                # 2. Key Step: Make ONLY the selected GPU visible to TensorFlow
                # This prevents TF from touching the memory of the other GPU.
                tf.config.set_visible_devices(physical_gpus[index], 'GPU')

                # 3. Set memory growth only on the visible device
                # Note: After setting visibility, we interact with the specific physical object
                tf.config.experimental.set_memory_growth(physical_gpus[index], True)

                log.info(f"Physically selected GPU: {physical_gpus[index].name}")
                log.info("Other GPUs are now invisible to TensorFlow.")

            except Exception as e:
                print("Error setting up GPU: {}".format(e))
        else:
            log.info("No GPUs found. Running on CPU.")
        
    prediction(pTrainedModel=args.trainedModel,
        pPredictionChromosomesFolders=args.predictionChromosomesFolders,
        pPredictionChromosomes=args.predictionChromosomes,
        pOutputFolder=args.outputFolder,
        pMultiplier=args.multiplier,
        pBinSize=args.binSize,
        pBatchSize=args.batchSize,
        pWindowSize=args.windowSize,
        pMatrixOutputName=args.matrixOutputName,
        pParameterOutputFile=args.parameterOutputFile,
        pSaveMemory=args.saveMemory,
        pNumberOfBatches=args.numberOfBatches,
        pScope=None, 
        pMode=args.mode)
