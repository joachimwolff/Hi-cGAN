import os
import numpy as np
from tensorflow import dtypes as tfdtypes
from scipy.sparse import save_npz, csr_matrix
from tqdm import tqdm
import concurrent.futures

from . import utils
from . import records


import logging
log = logging.getLogger(__name__)


class DataContainer():
    def __init__(self, chromosome, matrixFilePath, chromatinFolder, binSize=None):
        self.chromosome = str(chromosome)
        self.matrixFilePath = matrixFilePath
        self.chromatinFolder = chromatinFolder
        self.FactorDataArray = None
        self.nr_factors = None
        self.sparseHiCMatrix = None
        self.sequenceArray = None
        self.binSize = None
        if matrixFilePath is None: #otherwise it will be defined by the Hi-C matrix itself upon loading
            self.binSize = binSize
        self.factorNames = None
        self.prefixDict_factors = None
        self.prefixDict_matrix = None
        self.prefixDict_sequence = None
        self.chromSize_factors = None
        self.chromSize_matrix = None
        self.storedFeatures = None
        self.storedFiles = None
        self.windowSize = None
        self.flankingSize = None
        self.maximumDistance = None
        self.data_loaded = False

    def __loadFactorData(self, ignoreChromLengths=False, scaleFeatures=False, clampFeatures=False):
        #load chromatin factor data from bigwig files
        if self.chromatinFolder is None:
            return
        #ensure that binSizes for matrix (if given) and factors match
        if self.binSize is None:
            msg = "No binSize given; use a Hi-C matrix or explicitly specify binSize for the container"   
            raise TypeError(msg)
        ###load data for a specific chromsome
        #get the names of the bigwigfiles
        chromatinIsFolder = os.path.isdir(self.chromatinFolder)
        if not os.path.isdir(self.chromatinFolder) or not os.listdir(self.chromatinFolder):
            raise OSError(f"Chromatin folder '{self.chromatinFolder}' does not exist or is empty.")
        
        if chromatinIsFolder:
            bigwigFileList = utils.getBigwigFileList(self.chromatinFolder)
            bigwigFileList = sorted(bigwigFileList)
        else:
            bigwigFileList = sorted(self.chromatinFolder)
        if len(bigwigFileList) is None:
            msg = "Warning: folder {:s} does not contain any bigwig files"
            msg = msg.format(self.chromatinFolder)
            print(msg)
            return
        #check the chromosome name prefixes (e.g. "" or "chr") and sizes
        chromSizeList = []
        prefixDict_factors = dict()
        for bigwigFile in bigwigFileList:
            try:
                prefixDict_factors[bigwigFile] = utils.getChromPrefixBigwig(bigwigFile)
                chromname = prefixDict_factors[bigwigFile] + self.chromosome
                chromSizeList.append( utils.getChromSizesFromBigwig(bigwigFile)[chromname] )
            except Exception as e:
                msg = str(e) + "\n"
                msg += "Could not load data from bigwigfile {}".format(bigwigFile) 
                raise IOError(msg)
        #the chromosome lengths should be equal in all bigwig files
        if len(set(chromSizeList)) != 1 and not ignoreChromLengths:
            msg = "Invalid data. Chromosome lengths differ in bigwig files:"
            for i, filename in enumerate(bigwigFileList):
                msg += "{:s}: {:d}\n".format(filename, chromSizeList[i])
            raise IOError(msg)
        elif len(set(chromSizeList)) != 1 and ignoreChromLengths:
            chromSize_factors = min(chromSizeList)
        else:
            chromSize_factors = chromSizeList[0]
        #the chromosome lengths of matrices and bigwig files must be equal
        if self.chromSize_matrix is not None \
                and self.chromSize_matrix != chromSize_factors:
            msg = "Chrom lengths not equal between matrix and bigwig files\n"
            msg += "Matrix: {:d} -- Factors: {:d}".format(self.chromSize_matrix, chromSize_factors)
            raise IOError(msg)
        #load the data into memory now
        self.factorNames = [os.path.splitext(os.path.basename(name))[0] for name in bigwigFileList]
        self.nr_factors = len(self.factorNames)
        self.prefixDict_factors = prefixDict_factors
        self.chromSize_factors = chromSize_factors
        nr_bins = int( np.ceil(self.chromSize_factors / self.binSize) )
        self.FactorDataArray = np.empty(shape=(len(bigwigFileList),nr_bins))
        msg = "Loaded {:d} chromatin features from folder {:s}\n"
        msg = msg.format(self.nr_factors, self.chromatinFolder)
        featLoadedMsgList = [] #pretty printing for features loaded

        def process_bigwig_file(bigwigFile):
            chromname = self.prefixDict_factors[bigwigFile] + self.chromosome
            tmpArray = utils.binChromatinFactor(pBigwigFileName=bigwigFile,
                                                pBinSizeInt=self.binSize,
                                                pChromStr=chromname,
                                                pChromSize=self.chromSize_factors)
            if clampFeatures:
                tmpArray = utils.clampArray(tmpArray)
            if scaleFeatures:
                tmpArray = utils.scaleArray(tmpArray)
            return tmpArray

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_bigwig_file, bigwigFile) for bigwigFile in bigwigFileList]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                tmpArray = future.result()
                self.FactorDataArray[i] = tmpArray
                nr_nonzero_abs = np.count_nonzero(tmpArray)
                nr_nonzero_perc = nr_nonzero_abs / tmpArray.size * 100
                msg2 = "{:s} - min. {:.3f} - max. {:.3f} - nnz. {:d} ({:.2f}%)"
                msg2 = msg2.format(bigwigFileList[i], tmpArray.min(), tmpArray.max(), nr_nonzero_abs, nr_nonzero_perc)
                featLoadedMsgList.append(msg2)
        self.FactorDataArray = np.transpose(self.FactorDataArray)
        print(msg + "\n".join(featLoadedMsgList))
            
    def __loadMatrixData(self, scaleMatrix=False):
        #load Hi-C matrix from cooler file
        if self.matrixFilePath is None:
            return
        try:
            prefixDict_matrix = {self.matrixFilePath: utils.getChromPrefixCooler(self.matrixFilePath)}
            chromname = prefixDict_matrix[self.matrixFilePath] + self.chromosome
            chromsize_matrix = utils.getChromSizesFromCooler(self.matrixFilePath)[chromname]
            sparseHiCMatrix, binSize = utils.getMatrixFromCooler(self.matrixFilePath, chromname)
        except:
            msg = "Error: Could not load data from Hi-C matrix {:s}"
            msg = msg.format(self.matrixFilePath)
            raise IOError(msg)
        #scale to 0..1, if requested
        if scaleMatrix:
            sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)       
        #ensure that chrom sizes for matrix and factors are the same
        if self.chromSize_factors is not None and self.chromSize_factors != chromsize_matrix:
            msg = "Chromsize of matrix does not match bigwig files\n"
            msg += "Matrix: {:d} -- Bigwig files: {:d}"
            msg = msg.format(chromsize_matrix, self.chromSize_factors)
            raise IOError(msg)
        self.chromSize_matrix = chromsize_matrix
        #ensure that binSizes for matrix and factors (if given) match
        if self.binSize is None or self.binSize == binSize:
            self.binSize = binSize
            self.sparseHiCMatrix = sparseHiCMatrix
        elif self.binSize is not None and self.binSize != binSize:
            msg = "Matrix has wrong binSize\n"
            msg += "Matrix: {:d} -- Binned chromatin factors {:d}"
            msg = msg.format(binSize, self.binSize)
            raise IOError(msg)
        msg = "Loaded cooler matrix {:s}\n".format(self.matrixFilePath)
        msg += "chr. {:s}, matshape {:d}*{:d} -- min. {:d} -- max. {:d} -- nnz. {:d}"
        msg = msg.format(self.chromosome, self.sparseHiCMatrix.shape[0], self.sparseHiCMatrix.shape[1], int(self.sparseHiCMatrix.min()), int(self.sparseHiCMatrix.max()), self.sparseHiCMatrix.getnnz() )
        print(msg)
    
    def __unloadFactorData(self):
        #unload chromatin factor data to save memory, but do not touch metadata 
        self.FactorDataArray = None
        
    def __unloadMatrixData(self):
        #unload matrix data to save memory, but do not touch metadata
        self.sparseHiCMatrix = None

    def unloadData(self):
        #unload all data to save memory, but do not touch metadata
        self.__unloadFactorData
        self.__unloadMatrixData
        self.windowSize = None
        self.flankingSize = None
        self.maximumDistance = None
        self.data_loaded = False
    
    def loadData(self, windowSize, flankingSize=None, maximumDistance=None, scaleFeatures=False, clampFeatures=False, scaleTargets=False):
        if not isinstance(windowSize, int):
            msg = "windowSize must be integer"
            raise TypeError(msg)
        if isinstance(maximumDistance, int):
            maximumDistance = np.clip(maximumDistance, a_min=1, a_max=self.windowSize)
        self.__loadMatrixData(scaleMatrix=scaleTargets)
        self.__loadFactorData(scaleFeatures=scaleFeatures, clampFeatures=clampFeatures)
        self.windowSize = windowSize
        self.flankingSize = flankingSize
        self.maximumDistance = maximumDistance
        self.data_loaded = True

    def checkCompatibility(self, containerIterable):
        ret = []
        try:
           for container in containerIterable:
               ret.append(self.__checkCompatibility(container))
        except:
            ret = [self.__checkCompatibility(containerIterable)]
        return np.all(ret)
        
    def __checkCompatibility(self, container):
        if not isinstance(container, DataContainer):
            return False
        if not self.data_loaded or not container.data_loaded:
            return False
        #check if the same kind of data is available for all containers
        factorsOK = type(self.FactorDataArray) == type(container.FactorDataArray)
        matrixOK = type(self.sparseHiCMatrix) == type(container.sparseHiCMatrix)
        #check if windowSize, flankingSize and maximumDistance match
        windowSizeOK = self.windowSize == container.windowSize
        flankingSizeOK = self.flankingSize == container.flankingSize
        maximumDistanceOK = self.maximumDistance == container.maximumDistance
        log.debug("Factors: {:s} -- Matrix: {:s} -- windowSize: {:s} -- flankingSize: {:s} -- maximumDistance: {:s}".format(str(factorsOK), str(matrixOK), str(windowSizeOK), str(flankingSizeOK), str(maximumDistanceOK)))
        log.debug("Chromatin folder: {:s} -- Nr. factors: {:s}".format(str(self.chromatinFolder), str(self.nr_factors)))
        log.debug("Chromatin folder: {:s} -- Nr. factors: {:s}".format(str(container.chromatinFolder), str(container.nr_factors)))
        #sanity check loading of bigwig files
        if self.chromatinFolder is not None and self.nr_factors is None:
            return False
        if container.chromatinFolder is not None and container.nr_factors is None:
            return False
        #if chromatin factors are present, the numbers and names of chromatin factors must match
        factorsOK = factorsOK and (self.nr_factors == container.nr_factors)
        # factorsOK = factorsOK and (self.factorNames == container.factorNames)
        log.debug("self.nr_factors: {:d} -- container.nr_factors: {:d}".format(self.nr_factors, container.nr_factors))
        log.debug("self.factorNames: {:s} -- container.factorNames: {:s}".format(str(self.factorNames), str(container.factorNames)))
        return factorsOK and matrixOK and windowSizeOK and flankingSizeOK and maximumDistanceOK
        
    def writeTFRecord(self, pOutputFolder, pRecordSize=None):
        '''
        Write a dataset to disk in tensorflow TFRecord format
        
        Parameters:
            pwindowSize (int): size of submatrices
            pOutfolder (str): directory where TFRecords will be written
            pflankingSize (int): size of flanking regions left/right of submatrices
            pmaximumDistance (int): cut the matrices off at this distance (in bins)
            pRecordsize (int): split the TFRecords into multiple files containing approximately this number of samples
        
        Returns:
            list of filenames written
        '''

        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to write"
            print(msg)
            return None
        nr_samples = self.getNumberSamples()
        #adjust record size (yields smaller files and reduces memory load)
        recordsize = nr_samples
        if pRecordSize is not None and pRecordSize < recordsize:
            recordsize = pRecordSize
        #compute number of record files, number of samples 
        #in each file and corresponding indices
        nr_files = int( np.ceil(nr_samples/recordsize) )
        target_ct = int( np.floor(nr_samples/nr_files) )
        samples_per_file = [target_ct]*(nr_files-1) + [nr_samples-(nr_files-1)*target_ct]
        sample_indices = [sum(samples_per_file[0:i]) for i in range(len(samples_per_file)+1)] 
        #write the single files
        folderName = self.chromatinFolder.strip("/").replace("/","_")
        recordfiles = [os.path.join(pOutputFolder, "{:s}_{:s}_{:03d}.tfrecord".format(folderName, str(self.chromosome), i + 1)) for i in range(nr_files)]

        def storeTFRecord(recordfile, firstIndex, lastIndex, outfolder):
            log.debug("Prepare dict...")
            recordDict, storedFeaturesDict = self.__prepareWriteoutDict(pFirstIndex=firstIndex, 
                                                                        pLastIndex=lastIndex, 
                                                                        pOutfolder=outfolder)
            log.debug("Prepare dict... DONE!")
            log.debug("Write TFRecord...")
            records.writeTFRecord(pFilename=recordfile, pRecordDict=recordDict)
            log.debug("Write TFRecord... DONE!")

            return storedFeaturesDict

        storedFeaturesDictList = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(storeTFRecord, recordfile, firstIndex, lastIndex, pOutputFolder) for recordfile, firstIndex, lastIndex in zip(recordfiles, sample_indices, sample_indices[1:])]
            for future in concurrent.futures.as_completed(results):
                storedFeaturesDict = future.result()
                storedFeaturesDictList.append(storedFeaturesDict)
        self.storedFiles = recordfiles
        self.storedFeatures = storedFeaturesDict
        return recordfiles

    def getNumberSamples(self):
        if not self.data_loaded:
            return None
        featureArrays = [self.FactorDataArray, self.sparseHiCMatrix, self.sequenceArray]
        cutouts = [self.windowSize+2*self.flankingSize, self.windowSize+2*self.flankingSize, (self.windowSize+2*self.flankingSize)*self.binSize]
        nr_samples_list = []
        for featureArray, cutout in zip(featureArrays, cutouts):
            if featureArray is not None:
                nr_samples_list.append(featureArray.shape[0] - cutout + 1)
            else:
                nr_samples_list.append(0)
        #check if all features have the same number of samples
        if len(set( [x for x in nr_samples_list if x>0] )) != 1:
            msg = "Error: sample binning / DNA sequence encoding went wrong"
            raise RuntimeError(msg)
        return max(nr_samples_list)

    def __getMatrixData(self, idx):
        if self.matrixFilePath is None:
            return None # this can't work
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        #the 0-th matrix starts flankingSize away from the boundary
        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if flankingSize is None:
            flankingSize = windowSize
            self.flankingSize = windowSize
        startInd = idx + flankingSize
        stopInd = startInd + windowSize
        trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()
        trainmatrix = np.array(np.nan_to_num(trainmatrix))
        trainmatrix = np.expand_dims(trainmatrix, axis=-1) #make Hi-C (sub-)matrix an RGB image
        return trainmatrix
    
    def __getFactorData(self, idx):
        if self.chromatinFolder is None:
            return None
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        #the 0-th feature matrix starts at position 0
        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if flankingSize is None:
            flankingSize = windowSize
            self.flankingSize = windowSize
        startIdx = idx
        endIdx = startIdx + 2*flankingSize + windowSize
        factorArray = self.FactorDataArray[startIdx:endIdx]
        factorArray = np.expand_dims(factorArray, axis=-1)
        return factorArray

    def getSampleData(self, idx):
        if not self.data_loaded:
            return None
        factorArray = self.__getFactorData(idx)
        matrixArray = self.__getMatrixData(idx)
        if matrixArray is not None:
            matrixArray = matrixArray.astype("float32")
        return {"factorData": factorArray.astype("float32"), 
                "out_matrixData": matrixArray}
        
    def plotFeatureAtIndex(self, idx, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(idx, int) and (idx >= self.FactorDataArray.shape[0] or idx < 0):
            msg = "Error: Invalid index {:d}; must be None or integer in 0..{:d}".format(idx, self.FactorDataArray.shape[0]-1)
            raise ValueError(msg)
        if isinstance(idx, int):
            factorArray = self.__getFactorData(idx)
            startBin = idx
        else:
            factorArray = self.FactorDataArray 
            startBin = None
        for plotType in ["box", "line"]:   
            utils.plotChromatinFactors(pFactorArray=factorArray, 
                                        pFeatureNameList=self.factorNames,
                                        pChromatinFolder=self.chromatinFolder,
                                        pChrom=self.chromosome,
                                        pbinSize=self.binSize,
                                        pStartbin=startBin,
                                        pOutputPath=outpath,
                                        pPlotType=plotType,
                                        pFigureType=figuretype)
    
    def plotFeaturesAtPosition(self, position, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(position, int) and position > self.chromSize_factors:
            msg = "Error: Invalid position {:d}; must be in 0..{:d}"
            msg = msg.format(position, self.chromSize_factors)
            raise ValueError(msg)
        #compute the bin index from the position
        elif isinstance(position, int):
            idx = int(np.floor(position / self.binSize))
        else:
            idx = None
        return self.plotFeatureAtIndex(idx=idx,
                                        outpath=outpath,
                                        figuretype=figuretype)

    def saveMatrix(self, outputpath, index=None):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to save"
            print(msg)
            return
        sparseMatrix = None
        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if not isinstance(flankingSize, int):
            flankingSize = windowSize
        if isinstance(self.maximumDistance, int) and self.maximumDistance < windowSize and self.maximumDistance > 0:
            maximumDistance = self.maximumDistance
        else:
            maximumDistance = windowSize
        if isinstance(index, int) and index < self.getNumberSamples():
            tmpMat = np.zeros(shape=(windowSize, windowSize))
            indices = np.mask_indices(windowSize, utils.maskFunc, k=maximumDistance)
            tmpMat[indices] = self.__getMatrixData(idx=index)
            sparseMatrix = csr_matrix(tmpMat)
        else:
            sparseMatrix = self.sparseHiCMatrix
        folderName = self.chromatinFolder.rstrip("/").replace("/","-")
        filename = "matrix_{:s}_chr{:s}_{:s}".format(folderName, str(self.chromosome), str(index))
        filename = os.path.join(outputpath, filename)
        save_npz(file=filename, matrix=sparseMatrix)

    def __prepareWriteoutDict(self, pFirstIndex, pLastIndex, pOutfolder):
        if not self.data_loaded:
            msg = "Error: no data loaded, nothing to prepare"
            raise RuntimeError(msg)

        def get_sample_data(idx):
            return self.getSampleData(idx=idx)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = list(executor.map(get_sample_data, range(pFirstIndex, pLastIndex)))
        recordDict = dict()
        storedFeaturesDict = dict()
        if len(data) < 1:
            msg = "Error: No data to write"
            raise RuntimeError(msg)

        for key in data[0]:
            featData = [feature[key] for feature in data]
            if not any(elem is None for elem in featData):
                recordDict[key] = np.array(featData)
                storedFeaturesDict[key] = {"shape": recordDict[key].shape[1:], "dtype": tfdtypes.as_dtype(recordDict[key].dtype)}

        def process_feature(key):
            featData = [feature[key] for feature in data]
            if not any(elem is None for elem in featData):
                recordDict[key] = np.array(featData)
                storedFeaturesDict[key] = {"shape": recordDict[key].shape[1:], "dtype": tfdtypes.as_dtype(recordDict[key].dtype)}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_feature, data[0].keys())
        return recordDict, storedFeaturesDict