from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

import os
import math
import time
import simple_model
import inference_model
from simple_data_utils import Corpus # 必须要有，不然没法加载corpus对象
import simple_data_utils
from gpu_utils import assign_to_gpu, average_grads_and_vars
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
Do_Train = True
Do_Conditional_Inference = False
Do_eval = False

DATASET = "taobao"  # enwik8 wikitext-2 wikitext-103 taobao
print('the dataset is {}'.format(DATASET))

# Model
N_LAYER = 16
D_MODEL = 512   #  Dimension of the model 每一层输出维度？
D_EMBED = 512   # 词嵌入(字母嵌入)维度
N_HEAD = 10     # the number of multi-Head Attention 注意力头的个数 16模型崩了
D_HEAD = 41     # Dimension of each attention head 每个注意头包含注意力的个数
D_INNER = 1024  # Dimension of inner hidden size in positionwise feed-forward. feed-forward network 神经元节点的个数
NUM_CORE = 1

# Training
TGT_LEN = 100  # "Number of steps to predict"
MEM_LEN = 100  # Number of steps to cache
TRAIN_STEPS = 350000

BSZ = 32  # batch size

# # Testing
# TEST_TGT_LEN = 64
# TEST_MEM_LEN = 640
# TEST_CLAMP_LEN = 400
# TEST_BSZ = 10
# TEST_NUM_CORE = 1

# Eval
EVAL_BATCH_SIZE = BSZ
EVAL_NUM_CORE = NUM_CORE


# GPU config
flags.DEFINE_integer("num_hosts", default=1,
                     help="Number of GPU hosts")
flags.DEFINE_integer("num_core_per_host", default=NUM_CORE,
                     help="Number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_dir", default="data/{}/tfrecords".format(DATASET),
                    help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="data/{}/tfrecords".format(DATASET),
                    help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="data/{}/corpus-info.json".format(DATASET),
                    help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default='EXP-{}'.format(DATASET),
                    help="Estimator model_dir.")
flags.DEFINE_bool("do_train", default=Do_Train,
                  help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=Do_eval,
                  help="Whether to run eval on the dev set.")
flags.DEFINE_bool("do_con_inference", default=Do_Conditional_Inference,
                  help="Whether to conditional inference.")


flags.DEFINE_string("eval_ckpt_path", None,
                    help="Checkpoint path for do_test evaluation."
                         "If set, model_dir will be ignored."
                         "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_string("warm_start_path", None,
                    help="Checkpoint path for warm start."
                         "If set, will clear Adam states."
                         "Note that the new model_dir should be different"
                         " from warm_start_path.")

# Optimization config
flags.DEFINE_float("learning_rate", default=0.00025,
                   help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
                   help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
                   help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
                     help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size", default=BSZ,
                     help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=EVAL_BATCH_SIZE,
                     help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=TRAIN_STEPS,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=200,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=4000,
                     help="number of steps for model checkpointing.")
flags.DEFINE_string("dataset", default=DATASET,
                    help="Dataset name.")

# Evaluation config
flags.DEFINE_bool("do_test", default=False,
                  help="Run on the test set.")
flags.DEFINE_integer("max_eval_batch", default=-1,
                     help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False,
                  help="Run evaluation only.")
flags.DEFINE_integer("start_eval_steps", default=10000,
                     help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split", "valid",
                    help="Which data split to evaluate.")

# Inference config
flags.DEFINE_integer("inference_bsz", default=1,
                     help="inference batch size")


# Model config
flags.DEFINE_integer("tgt_len", default=TGT_LEN,
                     help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=MEM_LEN,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

flags.DEFINE_integer("n_layer", default=N_LAYER,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=D_MODEL,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=D_EMBED,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=N_HEAD,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=D_HEAD,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=D_INNER,
                     help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
                  help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
                  help="Tie embedding and softmax weight.")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
                  help="True to share all but first projs, False not to share.")
# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


def get_model_fn(n_token):
    def model_fn(inp, labels, mems, is_training):
        inp = tf.transpose(inp, [1, 0])
        labels = tf.transpose(labels, [1, 0])

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":              # select
            initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)

        tie_projs = [False]  # [False]
        if FLAGS.proj_share_all_but_first:  # 没用到，可能是TPU模式下才会用到
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True

        loss, new_mems = simple_model.transformer(
            dec_inp=inp,
            target=labels,
            mems=mems,
            n_token=n_token,
            n_layer=FLAGS.n_layer,
            d_model=FLAGS.d_model,
            d_embed=FLAGS.d_embed,
            n_head=FLAGS.n_head,
            d_head=FLAGS.d_head,
            d_inner=FLAGS.d_inner,
            dropout=FLAGS.dropout,
            dropatt=FLAGS.dropatt,
            initializer=initializer,
            proj_initializer=proj_initializer,
            is_training=is_training,
            mem_len=FLAGS.mem_len,
            same_length=FLAGS.same_length,
            clamp_len=FLAGS.clamp_len,
            untie_r=FLAGS.untie_r,)

        # number of parameters
        num_params = sum([int(np.prod(v.shape)) for v in tf.trainable_variables()]) # np.prod()计算所有元素的乘积
        tf.logging.info('#params: {}'.format(num_params))

        if is_training:
            all_vars = tf.trainable_variables()
            grads = tf.gradients(loss, all_vars)  #
            grads_and_vars = list(zip(grads, all_vars))  # [(grads[0],all_vars[1]),(grads[0],all_vars[1])]

            return loss, new_mems, grads_and_vars
        else:
            return loss, new_mems

    return model_fn


def inference_graph(n_token, inp, mems):
    # input由外部place_holder输入
    inp = tf.transpose(inp, [1, 0])

    if FLAGS.init == "uniform":
        initializer = tf.initializers.random_uniform(
            minval=-FLAGS.init_range,
            maxval=FLAGS.init_range,
            seed=None)
    elif FLAGS.init == "normal":  # select
        initializer = tf.initializers.random_normal(
            stddev=FLAGS.init_std,
            seed=None)
        proj_initializer = tf.initializers.random_normal(
            stddev=FLAGS.proj_init_std,
            seed=None)

    tie_projs = [False]  # [False]

    new_mems, logits_out = inference_model.transformer(
        dec_inp=inp,
        mems=mems,
        n_token=n_token,  # wrt. vocab_size used to build word embedding matrix
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        proj_initializer=proj_initializer,
        is_training=False,
        mem_len=FLAGS.mem_len,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        untie_r=FLAGS.untie_r)

    # number of parameters
    num_params = sum([int(np.prod(v.shape)) for v in tf.trainable_variables()])  # np.prod()计算所有元素的乘积
    tf.logging.info('#params: {}'.format(num_params))

    return new_mems, logits_out


def single_core_graph(n_token, is_training, inp, labels, mems):
    model_fn = get_model_fn(n_token=n_token)

    model_ret = model_fn(
        inp=inp,
        labels=labels,
        mems=mems,
        is_training=is_training)

    return model_ret


def train(n_token, ps_device):
    #  Get input function and model function
    train_input_fn, train_record_info = simple_data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="train",
        per_host_bsz=FLAGS.train_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=1)

    num_batches = train_record_info["num_batch"]
    tf.logging.info("num of batches {}".format(num_batches))
    tf.logging.info("run {} epochs:".format(TRAIN_STEPS / num_batches))

    # Create computational graph
    train_set = train_input_fn({
        "batch_size": FLAGS.train_batch_size,
        "data_dir": FLAGS.data_dir})

    input_feed, label_feed = train_set.make_one_shot_iterator().get_next()

    # 因为需要把一批数据分配到不同的机器上，需要将这批数据分割
    # tf.split(input, num_split, dimension) # num_split:份数，dimension：在哪个维度上切分，函数返回的列表(list)
    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)  # 第0个维度表示batch size
    labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

    per_core_bsz = FLAGS.train_batch_size // FLAGS.num_core_per_host

    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

    # assign_to_gpu(i, ps_device)
    for i in range(FLAGS.num_core_per_host):
        reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, ps_device)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            all_mems = [tf.placeholder(tf.float32,
                                       [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]  # all_mems，list类型，保存了每一层的mems

            loss_i, new_all_mems, grads_and_vars_i = single_core_graph(   # 为什么变量也要呢，有啥用？？？
                n_token=n_token,  # 字表字母的个数，27，26+1，a..z+_
                is_training=True,
                inp=inputs[i],
                labels=labels[i],
                mems=all_mems)

            tower_mems.append(all_mems)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_all_mems)
            tower_grads_and_vars.append(grads_and_vars_i)

    #  average losses and gradients across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]
    grads, all_vars = zip(*grads_and_vars)

    # clip gradient
    clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)  # 梯度张量和一个所有张量的全局范数
    grads_and_vars = list(zip(clipped, all_vars))

    # configure the optimizer
    global_step = tf.train.get_or_create_global_step()

    # warmup stage: increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
        warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                    * FLAGS.learning_rate
    else:
        warmup_lr = 0.0

    # decay stage: decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)

    # choose warmup or decay
    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    # get the train op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    # Training loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
            for layer in range(FLAGS.n_layer)] for core in range(FLAGS.num_core_per_host)
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.warm_start_path is not None:
            tf.logging.info("warm start from {}".format(FLAGS.warm_start_path))
            saver.restore(sess, FLAGS.warm_start_path)

        fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]

        total_loss, prev_step = 0., -1
        epoch = 0
        while True:
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, curr_step = fetched[:3]
            total_loss += loss_np

            if curr_step > 0 and curr_step % FLAGS.iterations == 0:
                curr_loss = total_loss / (curr_step - prev_step)
                tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
                                "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
                    curr_step, fetched[-3], fetched[-2],
                    curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
                total_loss, prev_step = 0., curr_step

            if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
                save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                saver.save(sess, save_path)
                tf.logging.info("Model saved in path: {}".format(save_path))

            if curr_step > 0 and curr_step % num_batches == 0:  # 整除一次，相当于一个epoch
                epoch += 1
                tf.logging.info("epoch: {} Done".format(epoch))
            if curr_step == FLAGS.train_steps:
                break

        tf.logging.info("run {} epochs:".format(TRAIN_STEPS/num_batches))


def evaluate(n_token, ps_device):
    #  Get input function and model function
    tf.logging.info("Now，starting evaluating mode")
    eval_input_fn, eval_record_info = simple_data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split=FLAGS.eval_split,
        per_host_bsz=FLAGS.eval_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=1)

    num_batch = eval_record_info["num_batch"]
    if FLAGS.max_eval_batch > 0:
        num_batch = FLAGS.max_eval_batch
    tf.logging.info("num of batches {}".format(num_batch))

    #  Create computational graph
    eval_set = eval_input_fn({
        "batch_size": FLAGS.eval_batch_size,
        "data_dir": FLAGS.data_dir})

    input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

    per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
    tower_mems, tower_losses, tower_new_mems = [], [], []

    for i in range(FLAGS.num_core_per_host):
        with tf.device(assign_to_gpu(i, ps_device)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            all_mems = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            loss_i, new_all_mems= single_core_graph(
                n_token=n_token,
                is_training=False,
                inp=inputs[i],
                labels=labels[i],
                mems=all_mems)

            tower_mems.append(all_mems)  # 当前的memory
            tower_losses.append(loss_i)
            tower_new_mems.append(new_all_mems) # 新的memory

    # sum losses across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
    else:
        loss = tower_losses[0]

    # Evaluation loop
    # 定义初始化memory，全0
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.eval_ckpt_path is None:
            eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        else:
            eval_ckpt_path = FLAGS.eval_ckpt_path
        tf.logging.info("Evaluate {}".format(eval_ckpt_path))
        saver.restore(sess, eval_ckpt_path)

        fetches = [loss, tower_new_mems, tf.size(label_feed)]

        format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
            len(str(num_batch)))

        total_loss, total_cnt = 0, 0
        for step in range(num_batch):
            if step % (num_batch // 10) == 0:
                tf.logging.info(format_str.format(step, num_batch))

            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, cnt_np = fetched[:3]
            total_loss += loss_np * cnt_np
            total_cnt += cnt_np

        avg_loss = total_loss / total_cnt
        tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
            avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


def condition_inference(n_token, ps_device):
    tf.logging.info("Now，starting conditional mode")
    inference_length = 100
    inference_bsz = FLAGS.inference_bsz

    # 读取保存的corpus，用于将把词转化为词索引。在必须inference阶段保证corpus已经存在！！！
    print("reading the saved corpus object")
    corpus = simple_data_utils.get_saved_corpus("data/taobao/")  # data path，dataset name
    print("read corpus Done")

    # raw_text = input(" Model prompt >>> ")
    raw_text = '我 喜欢 这件'
    raw_text = raw_text.split()
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input("Model prompt >>> ")  # 输入的raw_text需要在输入时分词，如"杭州 艾耕 科技"

    text_indices = np.array([corpus.vocab.get_indices(raw_text)])

    # 把索引对应生成语句
    generated_sentence = corpus.vocab.get_symbols(text_indices[0])
    print(generated_sentence)
    print(text_indices)

    # 搭建模型图,在单机上inference
    with tf.device(assign_to_gpu(0, ps_device)), tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        all_mems = [tf.placeholder(tf.float32, [FLAGS.mem_len, inference_bsz, FLAGS.d_model])
                    for _ in range(FLAGS.n_layer)]
        input_data = tf.placeholder(tf.int32, [inference_bsz, None], name='input_data')
        new_all_mems, logits_out = inference_graph(n_token=n_token, inp=input_data, mems=all_mems)
        all_mems_past = [np.zeros([FLAGS.mem_len, inference_bsz, FLAGS.d_model], dtype=np.float32)
             for layer in range(FLAGS.n_layer)]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.eval_ckpt_path is None:
            eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        else:
            eval_ckpt_path = FLAGS.eval_ckpt_path
        tf.logging.info("Evaluate {}".format(eval_ckpt_path))

        saver.restore(sess, eval_ckpt_path)
        print("restore ckpt Done")

        feed_dict = {input_data: text_indices}
        fetches = [new_all_mems, logits_out]

        for i in range(inference_length):
            for m, m_np in zip(all_mems, all_mems_past):
                feed_dict[m] = m_np
            all_mems_past, _logtits_out = sess.run(fetches, feed_dict=feed_dict) # _logtits_out：[第几个词，第几个句子,词表中各个词的概率]
            next_word_index = np.argmax(_logtits_out[-1], axis=1)  # [word_index]
            text_indices = np.concatenate([text_indices, [next_word_index]], axis=1)
            feed_dict[input_data] = text_indices

        for sentence in text_indices:
            print(corpus.vocab.get_symbols(sentence))
        # print(corpus.vocab.get_symbols(text_indices[0]))

        # for m, m_np in zip(all_mems, all_mems_past):
        #     feed_dict[m] = m_np
        # # _logtits_out:[batch_size, ]
        # all_mems_past, _logtits_out = sess.run(fetches, feed_dict=feed_dict)
        # next_word_index = np.argmax(_logtits_out[-1], axis=1) # [word_index]
        # print(_logtits_out)
        # print(next_word_index)
        # print(corpus.vocab.get_symbols(next_word_index))

        # text_indices = np.concatenate([text_indices, [next_word_index]], axis=1)

def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    corpus_info = simple_data_utils.get_corpus_info(FLAGS.corpus_info_path)

    # print(corpus_info)
    n_token = corpus_info["vocab_size"]  # 27 26个字母加另外一个字符？？？？

    # print(n_token)
    tf.logging.info("n_token {}".format(n_token))

    if FLAGS.do_train:
        train_start_time = time.time()
        train(n_token, "/gpu:0")
        train_end_time = time.time()
        tf.logging.info("training costs time {}:".format(train_end_time - train_start_time))

    if FLAGS.do_eval:
        eval_start_time = time.time()
        evaluate(n_token, "/gpu:0")
        eval_end_time = time.time()
        tf.logging.info("eval costs time {}:".format(eval_end_time - eval_start_time))

    if FLAGS.do_con_inference:
        condition_inference(n_token, "/gpu:0")


if __name__ == "__main__":
    tf.app.run()
