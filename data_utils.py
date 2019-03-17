# 该文件是论文源码，由于需要兼顾到TPU的特性，所以代码会变得十分冗余。

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from functools import partial

from collections import Counter, OrderedDict
import pickle
import json
import multiprocessing as mp

import numpy as np

from absl import flags
import tensorflow as tf
from vocabulary import Vocab

from os.path import exists
from os import makedirs
# from tensorflow.gfile import Exists as exists
# from tensorflow.gfile import MakeDirs as makedirs
# from tensorflow.gfile import Glob as glob

import pickle

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def _preprocess(shard, train, vocab, save_dir, cutoffs, bin_sizes, bsz, tgt_len,
                num_core_per_host, use_tpu, num_shuffle):
    file_names = []
    num_batch = 0

    path = train[shard]
    data_shard = vocab.encode_file(path, ordered=False, add_double_eos=True)

    for shuffle in range(num_shuffle):
        basename = "train-{:03d}-{:02d}".format(shard, shuffle)
        print("Processing shard {} shuffle {}".format(shard, shuffle))

        np.random.shuffle(data_shard)
        file_name, num_batch_shuffle = create_ordered_tfrecords(
            save_dir, basename, np.concatenate(data_shard), bsz, tgt_len,
            num_core_per_host, cutoffs, bin_sizes, use_tpu=use_tpu)
        file_names.append(file_name)
        num_batch += num_batch_shuffle

    return file_names, num_batch


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):

        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        if self.dataset in ["ptb", "wikitext-2", "enwik8", "text8"]:
            self.vocab.count_file(os.path.join(path, "train.txt"))
            self.vocab.count_file(os.path.join(path, "valid.txt"))
            self.vocab.count_file(os.path.join(path, "test.txt"))
        elif self.dataset in ["wikitext-103", "taobao"]:
            self.vocab.count_file(os.path.join(path, "train1.txt"))
            self.vocab.count_file(os.path.join(path, "valid1.txt"))
        elif self.dataset == "lm1b":
            train_path_pattern = os.path.join(
                path, "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled", "news.en-*")
            train_paths = glob(train_path_pattern)

            # the vocab will load from file when build_vocab() is called
            # for train_path in sorted(train_paths):
            #   self.vocab.count_file(train_path, verbose=True)

        self.vocab.build_vocab()

        if self.dataset in ["ptb", "wikitext-2", "wikitext-103", "taobao"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train1.txt"), ordered=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid1.txt"), ordered=True)
            self.test = self.vocab.encode_file(
                os.path.join(path, "test1.txt"), ordered=True)
        elif self.dataset in ["enwik8", "text8"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train.txt"), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=True, add_eos=False)
        elif self.dataset == "lm1b":
            self.train = train_paths
            valid_path = os.path.join(path, "valid.txt")
            test_path = valid_path
            self.valid = self.vocab.encode_file(
                valid_path, ordered=True, add_double_eos=True)
            self.test = self.vocab.encode_file(
                test_path, ordered=True, add_double_eos=True)

        if self.dataset in ["wikitext-103"]: # ,'taobao'
            self.cutoffs = [0, 20000, 40000, 200000] + [len(self.vocab)]  # 这个cutoffs到底是用来干嘛的
        elif self.dataset == "lm1b":
            self.cutoffs = [0, 60000, 100000, 640000] + [len(self.vocab)]
        else:
            self.cutoffs = []

    def convert_to_tfrecords(self, split, save_dir, bsz, tgt_len,
                             num_core_per_host, **kwargs):
        FLAGS = kwargs.get('FLAGS')

        file_names = []
        use_tpu = FLAGS.use_tpu and not (split == "test" and num_core_per_host == 1)

        if use_tpu:
            record_name = "record_info-{}.bsz-{}.tlen-{}.core-{}.json".format(
                split, bsz, tgt_len, num_core_per_host)
        else:
            record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
                split, bsz, tgt_len)

        record_info_path = os.path.join(save_dir, record_name)
        bin_sizes = None
        num_batch = None
        if self.dataset in ["ptb", "wikitext-2", "wikitext-103", "enwik8", "text8", "taobao"]:
            data = getattr(self, split)
            bin_sizes = get_bin_sizes(
                data, bsz // num_core_per_host, tgt_len, self.cutoffs)
            file_name, num_batch = create_ordered_tfrecords(
                save_dir, split, data, bsz, tgt_len, num_core_per_host,
                self.cutoffs, bin_sizes,
                num_passes=FLAGS.num_passes if split == 'train' and use_tpu else 1,
                use_tpu=use_tpu)
            file_names.append(file_name)
        elif self.dataset == "lm1b":
            bin_sizes = get_bin_sizes(
                self.valid, bsz // num_core_per_host, tgt_len, self.cutoffs)
            if split == "train":
                np.random.seed(123456)
                num_batch = 0

                if FLAGS.num_procs > 1:
                    _preprocess_wrapper = partial(_preprocess,
                                                  train=self.train, vocab=self.vocab, save_dir=save_dir,
                                                  cutoffs=self.cutoffs, bin_sizes=bin_sizes, bsz=bsz,
                                                  tgt_len=tgt_len, num_core_per_host=num_core_per_host,
                                                  use_tpu=use_tpu, num_shuffle=FLAGS.num_shuffle)

                    pool = mp.Pool(processes=FLAGS.num_procs)
                    results = pool.map(_preprocess_wrapper, range(len(self.train)))
                    for res in results:
                        file_names.extend(res[0])
                        num_batch += res[1]
                else:
                    for shard, path in enumerate(self.train):
                        data_shard = self.vocab.encode_file(path, ordered=False,
                                                            add_double_eos=True)

                        num_shuffle = FLAGS.num_shuffle

                        for shuffle in range(num_shuffle):
                            print("Processing shard {} shuffle {}".format(shard, shuffle))
                            basename = "train-{:03d}-{:02d}".format(shard, shuffle)
                            np.random.shuffle(data_shard)
                            file_name, num_batch_ = create_ordered_tfrecords(
                                save_dir, basename, np.concatenate(data_shard), bsz, tgt_len,
                                num_core_per_host,
                                self.cutoffs, bin_sizes, use_tpu=use_tpu)
                            file_names.append(file_name)
                            num_batch += num_batch_

            else:
                file_name, num_batch = create_ordered_tfrecords(
                    save_dir, split, getattr(self, split), bsz, tgt_len,
                    num_core_per_host,
                    self.cutoffs, bin_sizes, use_tpu=use_tpu)
                file_names.append(file_name)

        with open(record_info_path, "w") as fp:
            record_info = {
                "filenames": file_names,
                "bin_sizes": bin_sizes,
                "num_batch": num_batch
            }
            json.dump(record_info, fp)


def get_bin_sizes(data, batch_size, tgt_len, cutoffs, std_mult=[2.5, 2.5, 2.5]):
    """
      Note: the `batch_size` here should be per-core batch size
    """
    bin_sizes = []

    def _nearest_to_eight(x):  # so that it's faster on TPUs
        y = x - x % 8
        return y + 8 if x % 8 >= 4 else max(8, y)

    if cutoffs:
        num_batch = len(data) // batch_size // tgt_len

        data = data[:batch_size * num_batch * tgt_len] #
        data = data.reshape(batch_size, num_batch, tgt_len)

        tot = batch_size * tgt_len
        for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
            mask = (data >= left) * (data < right)
            percents = mask.astype(np.float64).sum(2).sum(0) / tot
            mean = np.mean(percents)
            std = np.std(percents)

            bin_size = int(math.ceil(tgt_len * batch_size * (mean + std_mult[b] * std)))
            bin_size = _nearest_to_eight(bin_size)
            bin_sizes.append(bin_size)

    return bin_sizes


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def batchify(data, batch_size, num_passes):
    """
      if use_tpu = True: num_passes > 1

      Since TPU training requires entire [bsz x tgt_len] chunks, it can discard
      as many as `bsz * tgt_len` tokens in training. When `bsz` and `tgt_len` are
      both large, as in the case of TPU training for Transformer-XL, the problem
      may lead to detectable performance drop.

      Here, we use multiple randomly shifted copies to deal with this problem.
    """
    if num_passes > 1:
        data_len = len(data)
        double_data = np.concatenate([data, data])
        data_list = []
        for i in range(num_passes):
            start = np.random.randint(0, data_len)
            data_list.append(double_data[start:start + data_len])
        data = np.concatenate(data_list)

    num_step = len(data) // batch_size
    data = data[:batch_size * num_step]
    data = data.reshape(batch_size, num_step)

    return data


def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len,
                             num_core_per_host, cutoffs=[], bin_sizes=[],
                             num_passes=1, use_tpu=False):
    if use_tpu:
        file_name = "{}.bsz-{}.tlen-{}.core-{}.tfrecords".format(
            basename, batch_size, tgt_len, num_core_per_host)
    else:
        file_name = "{}.bsz-{}.tlen-{}.tfrecords".format(
            basename, batch_size, tgt_len)

    save_path = os.path.join(save_dir, file_name)
    record_writer = tf.python_io.TFRecordWriter(save_path)

    batched_data = batchify(data, batch_size, num_passes)

    num_batch = 0
    # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
    for t in range(0, batched_data.shape[1] - 1, tgt_len):
        cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
        # drop the remainder if use tpu
        if use_tpu and cur_tgt_len < tgt_len:
            break
        if num_batch % 500 == 0:
            print("  processing batch {}".format(num_batch))
        for idx in range(batch_size):
            inputs = batched_data[idx, t:t + cur_tgt_len]
            labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]

            # features dict
            feature = {
                "inputs": _int64_feature(inputs),
                "labels": _int64_feature(labels),
            }

            # if len(cutoffs) > 0 and use_tpu:
            #     # validate `bin_sizes` and `cutoffs`
            #     assert len(cutoffs) - len(bin_sizes) == 2, \
            #         "len(cutoffs) - len(bin_sizes) != 2"
            #
            #     # mask for bin 0
            #     left, right = cutoffs[:2]
            #     inp_mask = ((inputs >= left) * (inputs < right)).astype(np.float32)
            #     tgt_mask = ((labels >= left) * (labels < right)).astype(np.float32)
            #
            #     feature["inp_mask"] = _float_feature(inp_mask)
            #     feature["tgt_mask"] = _float_feature(tgt_mask)
            #
            #     # refresh `inp_cnts` and `tgt_cnts` for each TPU core
            #     if idx % (batch_size // num_core_per_host) == 0:
            #         inp_cnts = [0] * len(bin_sizes)
            #         tgt_cnts = [0] * len(bin_sizes)
            #
            #     head_labels = np.copy(labels)
            #     inp_pos_per_bin, tgt_pos_per_bin = [], []
            #     for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
            #         inp_pos = np.where((inputs >= left) * (inputs < right))[0]
            #         tgt_pos = np.where((labels >= left) * (labels < right))[0]
            #         inp_pos_per_bin.append(inp_pos)
            #         tgt_pos_per_bin.append(tgt_pos)
            #
            #         head_labels[tgt_pos] = cutoffs[1] + b
            #
            #     feature["head_labels"] = _int64_feature(head_labels)
            #
            #     # permutation feature
            #     def _add_perm_feature(feature, pos_per_bin, cnts, prefix):
            #         for b, pos in enumerate(pos_per_bin):
            #             idx_tuple = []
            #             for p in pos:
            #                 if cnts[b] < bin_sizes[b]:
            #                     idx_tuple.append([p, cnts[b]])
            #                     cnts[b] += 1
            #                 else:
            #                     break
            #
            #             n_tup = len(idx_tuple)
            #             tup = np.array(idx_tuple).reshape(n_tup * 2)
            #
            #             feature["{}_cnt_{}".format(prefix, b)] = _int64_feature([n_tup])
            #             feature["{}_tup_{}".format(prefix, b)] = _int64_feature(tup)
            #
            #     _add_perm_feature(feature, inp_pos_per_bin, inp_cnts, "inp")
            #     _add_perm_feature(feature, tgt_pos_per_bin, tgt_cnts, "tgt")

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            record_writer.write(example.SerializeToString())

        num_batch += 1

    record_writer.close()
    print("Done writing {}. batches: {}".format(file_name, num_batch))

    return file_name, num_batch


def get_lm_corpus(data_dir, dataset):

    fn = os.path.join(data_dir, "cache1.pkl")
    print(fn)
    if exists(fn):
        print("Loading cached dataset...")
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)  #在macos上不能读，在Linux上能读
    else:
        print("Producing dataset...")
        kwargs = {}
        # if vocab file exist, use the file build
        if exists(os.path.join(FLAGS.data_dir, "vocab.txt")):
            kwargs['vocab_file'] = os.path.join(FLAGS.data_dir, "vocab.txt")
        if dataset in ["wikitext-103", "wikitext-2", "taobao"]:
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = False
        elif dataset == "ptb":
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = True
        elif dataset == "lm1b":
            kwargs["special"] = []
            kwargs["lower_case"] = False
            kwargs["vocab_file"] = os.path.join(data_dir, "1b_word_vocab.txt")
        elif dataset in ["enwik8", "text8"]:
            pass

        corpus = Corpus(data_dir, dataset, **kwargs)

        print("Saving dataset...")
        with open(fn, "wb") as fp:
            #pickle.dump(corpus, fp)  #, protocol=2
            pickle_dump(corpus, fn)

        corpus_info = {
            "vocab_size": len(corpus.vocab),
            "cutoffs": corpus.cutoffs,
            "dataset": corpus.dataset
        }
        with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
            json.dump(corpus_info, fp)

    return corpus


def main(unused_argv):
    del unused_argv  # Unused

    corpus = get_lm_corpus(FLAGS.data_dir, FLAGS.dataset)  # data path，dataset name

    save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
    if not exists(save_dir):
        makedirs(save_dir)

    # test mode
    if FLAGS.per_host_test_bsz > 0:
        corpus.convert_to_tfrecords("test", save_dir, FLAGS.per_host_test_bsz,
                                    FLAGS.tgt_len, FLAGS.num_core_per_host,
                                    FLAGS=FLAGS)
        return

    for split, batch_size in zip(
            ["train", "valid"],
            [FLAGS.per_host_train_bsz, FLAGS.per_host_valid_bsz]):

        if batch_size <= 0: continue
        print("Converting {} set...".format(split))
        corpus.convert_to_tfrecords(split, save_dir, batch_size, FLAGS.tgt_len,
                                    FLAGS.num_core_per_host, FLAGS=FLAGS)


def load_record_info(record_info_dir, split, per_host_bsz, tgt_len,
                     num_core_per_host, use_tpu):
    if use_tpu:
        record_name = "record_info-{}.bsz-{}.tlen-{}.core-{}.json".format(
            split, per_host_bsz, tgt_len, num_core_per_host)
    else:
        record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
            split, per_host_bsz, tgt_len)

    record_info_path = os.path.join(record_info_dir, record_name)
    with open(record_info_path, "r") as fp:
        record_info = json.load(fp)

    return record_info


def get_input_fn(record_info_dir, split, per_host_bsz, tgt_len,
                 num_core_per_host, num_hosts=1, use_tpu=False):
    """Creates input function."""

    record_info = load_record_info(record_info_dir, split, per_host_bsz, tgt_len,
                                   num_core_per_host, use_tpu=use_tpu)

    file_names = record_info["filenames"]
    bin_sizes = record_info["bin_sizes"]
    num_batch = record_info["num_batch"]

    tf.logging.info("[{}] File names {}".format(split, file_names))

    def input_fn(params):
        # per-core batch size
        per_core_bsz = params["batch_size"]

        # data_dir could be a remote path, e.g., a google storage url
        data_dir = params["data_dir"]

        def parser(record):
            # preprocess "inp_perm" and "tgt_perm"
            def _process_perm_feature(example, prefix):
                for b in range(len(bin_sizes)):
                    cnt = example.pop("{}_cnt_{}".format(prefix, b))[0]
                    tup = example.pop("{}_tup_{}".format(prefix, b))

                    tup = tf.reshape(
                        tf.sparse_tensor_to_dense(tup),
                        shape=[cnt, 2])

                    # tf.float32
                    perm = tf.sparse_to_dense(
                        sparse_indices=tup,
                        output_shape=[tgt_len, bin_sizes[b]],
                        sparse_values=1.0,
                        default_value=0.0)

                    example["{}_perm_{}".format(prefix, b)] = perm

            # whether allow the last batch with a potentially shorter length
            if use_tpu:
                record_spec = {
                    "inputs": tf.FixedLenFeature([tgt_len], tf.int64),
                    "labels": tf.FixedLenFeature([tgt_len], tf.int64),
                }
            else:
                record_spec = {
                    "inputs": tf.VarLenFeature(tf.int64),
                    "labels": tf.VarLenFeature(tf.int64),
                }

            # permutation related features
            if bin_sizes and use_tpu:
                # tf.float32
                record_spec["inp_mask"] = tf.FixedLenFeature([tgt_len], tf.float32)
                record_spec["tgt_mask"] = tf.FixedLenFeature([tgt_len], tf.float32)

                record_spec["head_labels"] = tf.FixedLenFeature([tgt_len], tf.int64)

                for b in range(len(bin_sizes)):
                    record_spec["inp_cnt_{}".format(b)] = tf.FixedLenFeature([1], tf.int64)
                    record_spec["inp_tup_{}".format(b)] = tf.VarLenFeature(tf.int64)
                    record_spec["tgt_cnt_{}".format(b)] = tf.FixedLenFeature([1], tf.int64)
                    record_spec["tgt_tup_{}".format(b)] = tf.VarLenFeature(tf.int64)

            # retrieve serialized example
            example = tf.parse_single_example(
                serialized=record,
                features=record_spec)

            # transform permutation tuples to permutation matrices
            if bin_sizes and use_tpu:
                _process_perm_feature(example, "inp")
                _process_perm_feature(example, "tgt")

            # cast int64 into int32
            # cast sparse to dense
            for key in list(example.keys()):
                val = example[key]
                if tf.keras.backend.is_sparse(val):
                    # val = tf.sparse.to_dense(val)
                    val = tf.sparse_tensor_to_dense(val)
                if val.dtype == tf.int64:
                    val = tf.to_int32(val)
                example[key] = val

            if use_tpu:
                return example
            else:
                return example["inputs"], example["labels"]
        file_paths = []
        for file_name in file_names:
            file_path = os.path.join(data_dir, file_name)
            file_paths.append(file_path)

        if split == "train":
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            if len(file_paths) > 1:
                dataset = dataset.shuffle(len(file_paths)).repeat()
                dataset = tf.data.TFRecordDataset(dataset)

            elif num_hosts > 1:
                host_id = params["context"].current_host
                # drop the remaining batches
                num_batch_per_host = num_batch // num_hosts

                my_start_sample_id = (host_id * num_batch_per_host * num_core_per_host *
                                      per_core_bsz)
                my_sample_num = num_batch_per_host * num_core_per_host * per_core_bsz
                dataset = tf.data.TFRecordDataset(dataset).skip(
                    my_start_sample_id).take(my_sample_num)
            else:
                # dataset = tf.data.TFRecordDataset(dataset)
                dataset = tf.data.TFRecordDataset(file_paths[0])

            dataset = dataset.map(parser).cache().repeat()
            dataset = dataset.batch(per_core_bsz)
            dataset = dataset.prefetch(num_core_per_host * per_core_bsz)
        else:
            # do not shuffle, repeat or cache in evaluation
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            # dataset = tf.data.TFRecordDataset(dataset)
            dataset = tf.data.TFRecordDataset(file_paths[0])
            dataset = dataset.map(parser)
            dataset = dataset.batch(per_core_bsz)
        return dataset

    if split == "train" and num_hosts > 1:
        record_info["num_batch"] = num_batch // num_hosts

    return input_fn, record_info


def get_corpus_info(corpus_info_path):
    with open(corpus_info_path, "r") as fp:
        corpus_info = json.load(fp)
    return corpus_info


if __name__ == "__main__":

    # Data
    DATASET_NAME = 'taobao'  # enwik8  wikitext-103 taobao
    DATA_ROOT = 'data/{}/'.format(DATASET_NAME)  # enwik8  wikitext-103 taobao

    # Model
    # N_LAYER = 12
    # D_MODEL = 512
    # D_EMBED = 512
    # N_HEAD = 8  #
    # D_HEAD = 64
    # D_INNER = 2048

    # Training
    TGT_LEN = 10
    # MEM_LEN = 512

    BSZ = 5
    NUM_CORE = 1

    # Testing
    # TEST_TGT_LEN = 80
    # TEST_MEM_LEN = 2100
    # TEST_CLAMP_LEN = 820

    TEST_BSZ = 10
    TEST_NUM_CORE = 1

    USE_TPU = False

    # If > 0, enter test mode and process test set only.Otherwise, process train and dev sets only
    PER_HOST_TEST_BSZ = 0

    FLAGS = flags.FLAGS

    flags.DEFINE_string("data_dir", DATA_ROOT,
                        help="Location of the data corpus")
    flags.DEFINE_string("dataset", default=DATASET_NAME,
                        help="Dataset name.")
    flags.DEFINE_integer("per_host_train_bsz", BSZ,
                         help="train batch size each host")
    flags.DEFINE_integer("per_host_valid_bsz", BSZ,
                         help="valid batch size each host")
    flags.DEFINE_integer("per_host_test_bsz", PER_HOST_TEST_BSZ,
                         help="If > 0, enter test mode and process test set only."
                              "Otherwise, process train and dev sets only.")
    flags.DEFINE_integer("tgt_len", TGT_LEN,
                         help="number of tokens to predict")
    flags.DEFINE_integer("max_batch", -1,
                         help="run in debug mode")
    flags.DEFINE_integer("num_core_per_host", NUM_CORE,
                         help="16 for dragonfish pod, 8 for dragonfish donut, 8 for jellyfish")
    flags.DEFINE_bool("debug", default=False,
                      help="Process only the first batch without shuffle for lm1b.")
    flags.DEFINE_integer("num_procs", 4,
                         help="number of processes")
    flags.DEFINE_integer("num_passes", 1,
                         help="number of passes when use_tpu=True")
    flags.DEFINE_integer("num_shuffle", 4,
                         help="number of shuffles for lm1b")
    flags.DEFINE_bool("use_tpu", USE_TPU,
                      help="use tpu")

    tf.app.run(main)
