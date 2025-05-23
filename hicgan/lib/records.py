import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import logging
log = logging.getLogger(__name__)


#parse serialized input to tensors
def parse_function(example_proto, descriptionDict):
    featDict = dict()
    targetDict = dict()
    contents = [key for key in descriptionDict]
    dtypes = [descriptionDict[key]["dtype"] for key in contents]
    shapes = [descriptionDict[key]["shape"] for key in contents]
    features = {key: tf.io.FixedLenFeature((), tf.string) for key in contents}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    for name, dtype, shape in zip(contents, dtypes, shapes):
        if name.startswith("out_"):
            targetDict[name] = tf.reshape( tf.io.decode_raw(parsed_features[name], dtype), shape)
        else:
            featDict[name] = tf.reshape( tf.io.decode_raw(parsed_features[name], dtype), shape)
    retList = []
    if len(featDict) > 0:
        retList.append(featDict)
    if len(targetDict) > 0:
        retList.append(targetDict)
    return tuple(retList)

# helper functions from tensorflow TFRecord docs
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#write tfRecord to disk
def writeTFRecord(pFilename: str, pRecordDict: dict):
    for key in pRecordDict:
        if not isinstance(pRecordDict[key], np.ndarray):
            return
    batches = set()
    for key in pRecordDict:
        batches.add(pRecordDict[key].shape[0])
    if len(batches) > 1:
        msg = "Batch sizes are not equal"
        raise ValueError(msg)
    
    with tf.io.TFRecordWriter(pFilename, options="GZIP") as writer:
        for i in range(list(batches)[0]):
            feature = dict()
            def process_feature(key):
                return key, _bytes_feature(pRecordDict[key][i].flatten().tostring())

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_feature, key) for key in pRecordDict]
                for future in futures:
                    key, value = future.result()
                    feature[key] = value
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def mirror_function(tensor1, tensor2):
    t1 = tf.reverse(tensor1, axis=[0])
    t2 = tf.transpose(tensor2, perm=(1,0,2))
    t2 = tf.image.rot90(t2, 2)
    return {"factorData": t1}, {"out_matrixData": t2}