# coding=utf-8
# Copyright 2020 The Google Research Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""BERT-joint baseline for TyDi v1.0.

 This code is largely based on the Natural Questions baseline from
 https://github.com/google-research/language/blob/master/language/question_answering/bert_joint/run_nq.py.

 The model uses special tokens to dealing with offsets between the original
 document content and the wordpieces. Here are examples:
 [ContextId=N] [Q]
 The presence of these special tokens requires overwriting some of the [UNUSED]
 vocab ids of the public BERT wordpiece vocabulary, similar to NQ baseline.

Overview:
  1. data.py: Responsible for deserializing the JSON and creating Pythonic data
       structures
     [ Usable by any ML framework / no TF dependencies ]

  2. tokenization.py: Fork of BERT's tokenizer that tracks byte offsets.
     [ Usable by any ML framework / no TF dependencies ]

  3. preproc.py: Calls tokenization and munges JSON into a format usable by
       the model.
     [ Usable by any ML framework / no TF dependencies ]

  4. tf_io.py: Tensorflow-specific IO code (reads `tf.Example`s from
       TF records). If you'd like to use your own favorite DL framework, you'd
       need to modify this; it's only about 200 lines.

  4. tydi_modeling.py: The core TensorFlow model code. **If you want to replace
       BERT with your own latest and greatest, start here!** Similarly, if
       you'd like to use your own favorite DL framework, this would be
       the only file that should require heavy modification; it's only about
       200 lines.

  5. postproc.py: Does postprocessing to find the answer, etc. Relevant only
     for inference.
     [ Usable by any ML framework / minimal tf dependencies ]

  6. run_tydi.py: The actual main driver script that uses all of the above and
       calls Tensorflow to do the main training and inference loops.
"""

import collections
import json
import os

from random import choices
from absl import logging
from bert import modeling as bert_modeling
import tensorflow.compat.v1 as tf
import postproc
import preproc
import tf_io
import tydi_modeling
from bert import optimization as opt


import tensorflow.contrib as tf_contrib
tf.enable_eager_execution()
#LANGUAGE_LST = [103829, 53656, 54602, 42857, 84324, 33796, 49595, 73302, 95780, 80215, 37846]
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_records_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_string(
    "record_count_file", None,
    "File containing number of precomputed training records "
    "(in terms of 'features', meaning slices of articles). "
    "This is used for computing how many steps to take in "
    "each fine tuning epoch.")

flags.DEFINE_integer(
    "candidate_beam", 30,
    "How many wordpiece offset to be considered as boundary at inference time.")

flags.DEFINE_string(
    "predict_file", None,
    "TyDi json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz. "
    "Used only for `--do_predict`.")

flags.DEFINE_string(
    "precomputed_predict_file", None,
    "TyDi tf.Example records for predictions, created separately by "
    "`prepare_tydi_data.py` Used only for `--do_predict`.")

flags.DEFINE_string(
    "output_prediction_file", None,
    "Where to print predictions in TyDi prediction format, to be passed to"
    "tydi_eval.py.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained mBERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_question_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run prediction.")

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_integer(
    "predict_file_shard_size", 1000,
    "The maximum number of examples to put into each temporary TF example file "
    "used as model input a prediction time.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal TyDi evaluation.")

flags.DEFINE_integer(
    "max_passages", 45, "Maximum number of passages to consider for a "
    "single article. If an article contains more than"
    "this, they will be discarded during training. "
    "BERT's WordPiece vocabulary must be modified to include "
    "these within the [unused*] vocab IDs.")

flags.DEFINE_integer(
    "max_position", 45,
    "Maximum passage position for which to generate special tokens.")

flags.DEFINE_bool(
    "fail_on_invalid", True,
    "Stop immediately on encountering an invalid example? "
    "If false, just print a warning and skip it.")

flags.DEFINE_integer(
    "M", 16,
    "Amount of data to train the multilingual model before updating phi"
)

flags.DEFINE_float(
    "socrer_learning_rate", 5e-5,
    "The learning rate for updating scorer function"
)

flags.DEFINE_integer(
    "num_dev_sets", 2,
    "How many dev sets to use for averaging out MultiDDS scores"
)
flags.DEFINE_float(
    "train_size", 0.9,
    "How many percentage of dataset is split for train dataset"
)

### TPU-specific flags:

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_records_file:
      raise ValueError("If `do_train` is True, then `train_records_file` "
                       "must be specified.")
    if not FLAGS.record_count_file:
      raise ValueError("If `do_train` is True, then `record_count_file` "
                       "must be specified.")

  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError("If `do_predict` is True, "
                       "then `predict_file` must be specified.")
    if not FLAGS.output_prediction_file:
      raise ValueError("If `do_predict` is True, "
                       "then `output_prediction_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_question_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_question_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_question_length))


def main(_):

  logging.set_verbosity(logging.INFO)
  bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  validate_flags_or_throw(bert_config)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf_contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf_contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf_contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf_contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  '''if FLAGS.do_train:
    with tf.gfile.Open(FLAGS.record_count_file, "r") as f:
      num_train_features = int(f.read().strip())
    num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                          FLAGS.num_train_epochs)
    logging.info("record_count_file: %s", FLAGS.record_count_file)
    logging.info("num_records (features): %d", num_train_features)
    logging.info("num_train_epochs: %d", FLAGS.num_train_epochs)
    logging.info("train_batch_size: %d", FLAGS.train_batch_size)
    logging.info("num_train_steps: %d", num_train_steps)

    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = tydi_modeling.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this falls back to normal Estimator on CPU or GPU.
  estimator = tf_contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)'''

  if FLAGS.do_train:

    with tf.gfile.Open(FLAGS.record_count_file, "r") as f:
      num_train_features = int(f.read().strip())
    num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                          FLAGS.num_train_epochs)
    logging.info("record_count_file: %s", FLAGS.record_count_file)
    logging.info("num_records (features): %d", num_train_features)
    logging.info("num_train_epochs: %d", FLAGS.num_train_epochs)
    logging.info("train_batch_size: %d", FLAGS.train_batch_size)
    logging.info("num_train_steps: %d", num_train_steps)

    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    logging.info("Running training on precomputed features")
    logging.info("  Num split examples = %d", num_train_features)
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    logging.info("  Num steps = %d", num_train_steps)
    train_filenames = tf.gfile.Glob(FLAGS.train_records_file)

    model_fn = tydi_modeling.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        # This needs to be kept in sync with `FeatureWriter`.
        name_to_features = {
            "language_id": tf.FixedLenFeature([], tf.int64),
            "unique_ids": tf.FixedLenFeature([], tf.int64),
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        }

        if is_training:
            name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
            name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
            name_to_features["answer_types"] = tf.FixedLenFeature([], tf.int64)

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                    example[name] = t

            return example

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        d = d.shuffle(buffer_size=100)
        d = d.map(lambda record: _decode_record(record, name_to_features))
        return d

    def split_train_dev(dataset, train_port, num_dev):
        train_size = int(DATASET_SIZE * train_port)
        sub_dev_size = int(DATASET_SIZE * (1 - train_port) / num_dev)

        full_dataset = dataset.shuffle(100)
        train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)

        shard_devs = []
        for i in range(num_dev):
            shard_devs.append(test_dataset.shard(num_shards=num_dev, index=i))
        return train_dataset, shard_devs

    def count_dataset(dataset):
        cnt = 0
        for i in dataset.repeat(1).make_one_shot_iterator():
            # if cnt % 2000==0:
            # print(cnt)
            cnt += 1

        return cnt

    def split_langs(dataset):
        def dataset_fn(ds, i):
            return ds.filter(lambda x: tf.equal(x['language_id'], i))

        data_set_lst = []
        for i in range(11):
            dataset_filter_lang = dataset.apply(lambda x: dataset_fn(x, i))
            data_set_lst.append(dataset_filter_lang)
        return data_set_lst

    DATASET_SIZE = num_train_features


    NUM_LANGS = 11

    tf_dataset = input_fn_builder(FLAGS.train_records_file, 512, True, False)

    train_set, dev_shards = split_train_dev(tf_dataset, FLAGS.train_size, FLAGS.num_dev_sets)
    total_num_train_samples = FLAGS.train_size * DATASET_SIZE
    total_num_dev_samples = DATASET_SIZE - total_num_train_samples


    logging.info("Nums of examples in train dataset = %d", total_num_train_samples)
    logging.info("Total numbers of examples in dev set = %d", total_num_dev_samples)

    train_set_langs = split_langs(train_set)
    dev_set_langs = []

    for div_set in dev_shards:
        dev_set_langs.append(split_langs(div_set))

    #train_set_langs is a 1d lst, and div_set_langs is a 2d lst, notice that you could find
    #corresponding languages ids in data file

    def sample_lang_id(lang_freq):
        #print(lang_freq)
        #print(list(range(NUM_LANGS)))
        return choices(list(range(NUM_LANGS)), lang_freq)

    # count number of languages in each language

    lang_sample_dist = []
    for lang in train_set_langs:
        lang_cnt = count_dataset(lang)
        print(lang_cnt)
        lang_sample_dist.append(lang_cnt / total_num_train_samples)

    train_samplers = list(map(lambda x: x.repeat().batch(1).make_one_shot_iterator(), train_set_langs))

    dev_samplers = []
    for dev_set in dev_set_langs:
        dev_samplers.append(list(map(lambda x: iter(x.repeat().batch(1).make_one_shot_iterator()), dev_set)))

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=FLAGS.learning_rate, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    #uncomment this to enable warm-up steps
    '''if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = FLAGS.learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)'''

    optimizer = opt.AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    if FLAGS.use_tpu:
        optimizer = contrib_tpu.CrossShardOptimizer(optimizer)


    #do MultiDDS training
    #initialize sample distribution phis
    phi = tf.get_variable(
        "phi", [11], initializer=tf.truncated_normal_initializer(stddev=0.02))

    opt_scorer = tf.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
    name='Adam'
    )

    while global_step < num_train_steps:
        if not tf.equal(global_step, 0):
            lang_sample_dist = list(tf.nn.softmax(phi).numpy())

        #load training data with phi
        logging.info('We are sampling from train data')
        data_lst = []
        while len(data_lst) < FLAGS.M:
            #choose a langue to sample
            cur_lang = sample_lang_id(lang_sample_dist)
            data_lst.append(train_samplers[cur_lang[0]].get_next())

        logging.info('Train mBert for multiple steps')
        for data in data_lst:
            with tf.GradientTape() as tape:
                tvars, loss = model_fn(data, _, tf.estimator.ModeKeys.TRAIN, _, global_step)
                #print(loss)

            grads = tape.gradient(loss, tvars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
            optimizer.apply_gradients(
                zip(grads, tvars), global_step=global_step)

        logging.info('Estimate the effect of each language')
        rewards = []
        for i in range(NUM_LANGS):
            gradient_dev = 0
            gradient_train = 0
            #Some languages might not have samples
            try:
                train_test = train_samplers[i].get_next()
                with tf.GradientTape() as tape:
                    tvars, loss = model_fn(data, _, tf.estimator.ModeKeys.TRAIN, _, global_step)

                grads = tape.gradient(loss, tvars)
                gradient_train = grads
                #Not sure whether to add this line or not
                #TODO: modify me to allow functions
                (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
                optimizer.apply_gradients(
                    zip(grads, tvars), global_step=global_step)

                logging.info("Testing effect on other languages")


                for k in range(len(dev_samplers)):
                    for j in range(NUM_LANGS):
                        try:
                            dev_data = dev_samplers[k][j].get_next()
                            with tf.GradientTape() as tape:
                                tvars, loss = model_fn(data, _, tf.estimator.ModeKeys.TRAIN, _, global_step)
                            grads = tape.gradient(loss, tvars)
                            gradient_dev += grads

                        except:
                            print(j, 'language not exist in dataset', k)
            except:
                print("No data in this train language!!!")

            #append scores of each language to reward list
            print(gradient_train, gradient_dev)
            normalize_a = tf.nn.l2_normalize(gradient_dev, 0)
            normalize_b = tf.nn.l2_normalize(gradient_train, 0)
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
            rewards.append(cos_similarity)

        logging.info("Optimize phi!")
        grad_phi = 0
        for i in range(NUM_LANGS):
            log_i = tf.log(tf.nn.softmax(phi))[i]
            with tf.GradientTape() as tape:
                grads = tape.gradient(log_i, phi)
            grad_phi += grads * rewards[i]
        opt_scorer.apply_gradient(
            zip(grad_phi, phi), global_step=global_step
        )

        new_global_step = global_step + 1
        global_step.assign(new_global_step)


    #estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    if not FLAGS.precomputed_predict_file:
      predict_examples_iter = preproc.read_tydi_examples(
          input_file=FLAGS.predict_file,
          is_training=False,
          max_passages=FLAGS.max_passages,
          max_position=FLAGS.max_position,
          fail_on_invalid=FLAGS.fail_on_invalid,
          open_fn=tf_io.gopen)
      shards_iter = write_tf_feature_files(predict_examples_iter)
    else:
      # Uses zeros for example and feature counts since they're unknown, and
      # we only use them for logging anyway.
      shards_iter = enumerate(
          ((f, 0, 0) for f in tf.gfile.Glob(FLAGS.precomputed_predict_file)), 1)

    # Accumulates all of the prediction results to be written to the output.
    full_tydi_pred_dict = {}
    total_num_examples = 0
    for shard_num, (shard_filename, shard_num_examples,
                    shard_num_features) in shards_iter:
      total_num_examples += shard_num_examples
      logging.info(
          "Shard %d: Running prediction for %s; %d examples, %d features.",
          shard_num, shard_filename, shard_num_examples, shard_num_features)

      # Runs the model on the shard and store the individual results.
      # If running predict on TPU, you will need to specify the number of steps.
      predict_input_fn = tf_io.input_fn_builder(
          input_file=[shard_filename],
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=False)
      all_results = []
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True):
        if len(all_results) % 10000 == 0:
          logging.info("Shard %d: Predicting for feature %d/%s", shard_num,
                       len(all_results), shard_num_features)
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        answer_type_logits = [
            float(x) for x in result["answer_type_logits"].flat
        ]
        all_results.append(
            tydi_modeling.RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                answer_type_logits=answer_type_logits))

      # Reads the prediction candidates from the (entire) prediction input file.
      candidates_dict = read_candidates(FLAGS.predict_file)
      predict_features = [
          tf.train.Example.FromString(r)
          for r in tf.python_io.tf_record_iterator(shard_filename)
      ]
      logging.info("Shard %d: Post-processing predictions.", shard_num)
      logging.info("  Num candidate examples loaded (includes all shards): %d",
                   len(candidates_dict))
      logging.info("  Num candidate features loaded: %d", len(predict_features))
      logging.info("  Num prediction result features: %d", len(all_results))
      logging.info("  Num shard features: %d", shard_num_features)

      tydi_pred_dict = postproc.compute_pred_dict(
          candidates_dict,
          predict_features, [r._asdict() for r in all_results],
          candidate_beam=FLAGS.candidate_beam)

      logging.info("Shard %d: Post-processed predictions.", shard_num)
      logging.info("  Num shard examples: %d", shard_num_examples)
      logging.info("  Num post-processed results: %d", len(tydi_pred_dict))
      if shard_num_examples != len(tydi_pred_dict):
        logging.warning("  Num missing predictions: %d",
                        shard_num_examples - len(tydi_pred_dict))
      for key, value in tydi_pred_dict.items():
        if key in full_tydi_pred_dict:
          logging.warning("ERROR: '%s' already in full_tydi_pred_dict!", key)
        full_tydi_pred_dict[key] = value

    logging.info("Prediction finished for all shards.")
    logging.info("  Total input examples: %d", total_num_examples)
    logging.info("  Total output predictions: %d", len(full_tydi_pred_dict))

    with tf.gfile.Open(FLAGS.output_prediction_file, "w") as output_file:
      for prediction in full_tydi_pred_dict.values():
        output_file.write((json.dumps(prediction) + "\n").encode())


def write_tf_feature_files(tydi_examples_iter):
  """Converts TyDi examples to features and writes them to files."""
  logging.info("Converting examples started.")

  total_feature_count_frequencies = collections.defaultdict(int)
  total_num_examples = 0
  total_num_features = 0
  for shard_num, examples in enumerate(
      sharded_iterator(tydi_examples_iter, FLAGS.predict_file_shard_size), 1):
    features_writer = tf_io.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir,
                              "features.tf_record-%03d" % shard_num),
        is_training=False)
    num_features_to_ids, shard_num_examples = (
        preproc.convert_examples_to_features(
            tydi_examples=examples,
            vocab_file=FLAGS.vocab_file,
            is_training=False,
            max_question_length=FLAGS.max_question_length,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            include_unknowns=FLAGS.include_unknowns,
            output_fn=features_writer.process_feature))
    features_writer.close()

    if shard_num_examples == 0:
      continue

    shard_num_features = 0
    for num_features, ids in num_features_to_ids.items():
      shard_num_features += (num_features * len(ids))
      total_feature_count_frequencies[num_features] += len(ids)
    total_num_examples += shard_num_examples
    total_num_features += shard_num_features
    logging.info("Shard %d: Converted %d input examples into %d features.",
                 shard_num, shard_num_examples, shard_num_features)
    logging.info("  Total so far: %d input examples, %d features.",
                 total_num_examples, total_num_features)
    yield (shard_num, (features_writer.filename, shard_num_examples,
                       shard_num_features))

  logging.info("Converting examples finished.")
  logging.info("  Total examples = %d", total_num_examples)
  logging.info("  Total features = %d", total_num_features)
  logging.info("  total_feature_count_frequencies = %s",
               sorted(total_feature_count_frequencies.items()))


def sharded_iterator(iterator, shard_size):
  """Returns an iterator of iterators of at most size `shard_size`."""
  exhaused = False
  while not exhaused:

    def shard():
      for i, item in enumerate(iterator, 1):
        yield item
        if i == shard_size:
          return
      nonlocal exhaused
      exhaused = True

    yield shard()


def read_candidates(input_pattern):
  """Read candidates from an input pattern."""
  input_paths = tf.gfile.Glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    file_obj = tf_io.gopen(input_path)
    final_dict.update(postproc.read_candidates_from_one_split(file_obj))
  return final_dict


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
