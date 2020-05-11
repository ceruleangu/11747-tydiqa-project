#Run this command to start training

'''python run_tydi_torch.py --vocab_file=mbert_modified_vocab.txt \
                                     --init_checkpoint='bert-base-multilingual-cased' \
                                     --train_records_file=preprocessed_data/cached_train \
                                    --do_train=True \
                                               --output_dir ='tydiqa_baseline_model'''

'''python3 run_tydi_torch.py \
  --vocab_file='mbert_modified_vocab.txt' \
  --pretrained_weights='=tydiqa_baseline_model/checkpoint-317433' \
  --predict_file='tydiqa-v1.0-dev.jsonl.gz' \
  --precomputed_predict_file='dev_shards/*.tfrecord' \
  --do_predict=True\
  --do_train=False\
  --output_dir='~/tydiqa_baseline_model/predict' \
  --output_prediction_file='~/tydiqa_baseline_model/predict/pred3.jsonl'''

import gzip
import collections
import json
import os

import logging
import tensorflow.compat.v1 as tf
import postproc
from tydi_modeling_torch import TYDIQA
import torch
import argparse
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
#from tydi_modeling_torch import TYDIQA
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import tqdm
from tfrecord.torch.dataset import TFRecordDataset

from torch.utils.tensorboard import SummaryWriter




def train(args, train_dataset, model):
    # use tensorboard to keep track of training process
    tb_writer = SummaryWriter('loss')

    #train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle = True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataloader)
        * args.num_train_epochs
    )

    # Train!
    logging.info("***** Running training *****")
    #logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)

    logging.info("  Let's start finetuning!")
    tr_loss, logging_loss = 0.0, 0.0
    global_step = 0
    for epoch in tqdm.trange(args.num_train_epochs, desc = 'Epoch'):
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
            model.train()

            outputs = model(
                is_training = True,
                input_ids=batch["input_ids"].long().to(DEVICE),
                attention_mask=batch['input_mask'].long().to(DEVICE),
                token_type_ids=batch['segment_ids'].long().to(DEVICE),
                start_positions=batch["start_positions"].long().to(DEVICE),
                end_positions=batch["end_positions"].long().to(DEVICE),
                answer_types=batch["answer_types"].long().to(DEVICE)
            )

            loss = outputs[-1]
            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    print('loss', (tr_loss - logging_loss) / args.logging_steps)
                    logging_loss = tr_loss
            #empty cahce
            del batch
            torch.cuda.empty_cache()
            #loggin points


        # save checkpoint
        #if global_step % args.save_steps == 0:
        output_dir = os.path.join(args.output_dir, "checkpoint-3{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logging.info("Saving model checkpoint to %s", output_dir)

                #torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #logging.info("Saving optimizer and scheduler states to %s", output_dir)

    return global_step, tr_loss / global_step

# This represents the raw predictions coming out of the neural model.
RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])

def gopen(path):
  """Opens a file object given a (possibly gzipped) `path`."""
  logging.info("*** Loading from: %s ***", path)
  if ".gz" in path:
    return gzip.GzipFile(fileobj=tf.gfile.Open(path, "rb"))  # pytype: disable=wrong-arg-types
  else:
    return tf.gfile.Open(path, "r")
def read_candidates(input_pattern):
  """Read candidates from an input pattern."""
  input_paths = tf.gfile.Glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    file_obj = gopen(input_path)
    final_dict.update(postproc.read_candidates_from_one_split(file_obj))
  return final_dict

def predict(args, model):
    tf.gfile.MakeDirs(args.output_dir)
    #read prediction candidates from entire prediction input jsonl.gz file
    candidates_dict = read_candidates(args.predict_file)
    #Prediction!
    logging.info("start predicting!")

    #define following routines to call
    full_tydi_pred_dict = {}
    total_num_examples = 0
    shards_iter = enumerate(
        ((f, 0, 0) for f in sorted(tf.gfile.Glob(args.precomputed_predict_file))), 1)

    #Iterating through different shards to get results as we want
    for shard_num, (shard_filename, shard_num_examples,
                    shard_num_features) in shards_iter:
        all_results = []
        total_num_examples += shard_num_examples
        logging.info(
            "Shard %d: Running prediction for %s; %d examples, %d features.",
            shard_num, shard_filename, shard_num_examples, shard_num_features
        )
        print(shard_filename)
        #use tfrecord_dataset to read tfrecord into dataset for pytorch
        eval_dataset = TFRecordDataset(shard_filename, index_path = None)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.predict_batch_size, shuffle=False)

        for step, batch in enumerate(eval_dataloader):
            #Turn on model evaluation mode, and set frame to no_grad
            model.eval()
            with torch.no_grad():
                outputs = model(
                    is_training=False,
                    input_ids=batch["input_ids"].long().to(DEVICE),
                    attention_mask=batch['input_mask'].long().to(DEVICE),
                    token_type_ids=batch['segment_ids'].long().to(DEVICE),
                )
                #print(torch.max(outputs[0], -1))
                #write results into RawResult format for post-process
                for num, (i, j, k) in enumerate(zip(outputs[0], outputs[1], outputs[2])):
                    unique_ids = int(batch['unique_ids'][num])
                    start_logits = [float(x) for x in i]
                    end_logits = [float(x) for x in j]
                    answer_type_logits = [float(x) for x in k]
                    all_results.append(
                        RawResult(
                            unique_id=unique_ids,
                            start_logits=start_logits,
                            end_logits=end_logits,
                            answer_type_logits=answer_type_logits
                        )
                    )
            print('We at step %d of shard % d', step, shard_filename)

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

        #pass candidates dict, raw results and features to postproc for later use
        tydi_pred_dict = postproc.compute_pred_dict(
            candidates_dict,
            predict_features, [r._asdict() for r in all_results],
            candidate_beam=args.candidate_beam)

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
        #break
    #Finish up predictions for all shards and start logging
    logging.info("Prediction finished for all shards.")
    logging.info("  Total input examples: %d", total_num_examples)
    logging.info("  Total output predictions: %d", len(full_tydi_pred_dict))

    with tf.gfile.Open(args.output_prediction_file, "w") as output_file:
        for prediction in full_tydi_pred_dict.values():
            output_file.write((json.dumps(prediction) + "\n").encode())


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(filename='tydi.log', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Arguments for running tydi')

    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=False,
        help="The config json file corresponding to the pre-trained BERT model. "
             "This specifies the model architecture."
    )

    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        required=False,
        help="The vocabulary file that the BERT model was trained on."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."

    )

    parser.add_argument(
        "--train_records_file",
        default=None,
        type=str,
        required=False,
        help="Precomputed tf records for training."

    )

    parser.add_argument(
        "--record_count_file",
        default=None,
        type=str,
        required=False,
        help="File containing number of precomputed training records "
             "(in terms of 'features', meaning slices of articles). "
             "This is used for computing how many steps to take in "
             "each fine tuning epoch."

    )

    parser.add_argument(
        "--candidate_beam",
        default=30,
        type=int,
        required=False,
        help="How many wordpiece offset to be considered as boundary at inference time."

    )

    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        required=False,
        help="TyDi json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz. "
             "Used only for `--do_predict`."
    )

    parser.add_argument(
        "--precomputed_predict_file",
        default=None,
        type=str,
        required=False,
        help="TyDi json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz. "
             "Used only for `--do_predict`."
    )

    parser.add_argument(
        "--output_prediction_file",
        default=None,
        type=str,
        required=False,
        help="Where to print predictions in TyDi prediction format, to be passed to"
             "tydi_eval.py."
    )

    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        required=False,
        help="Initial checkpoint (usually from a pre-trained mBERT model)."
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        required=False,
        help="Where to print predictions in TyDi prediction format, to be passed to"
             "tydi_eval.py."
    )

    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        required=False,
        help="When splitting up a long document into chunks, how much stride to "
             "take between chunks."
    )

    parser.add_argument(
        "--max_question_length",
        default=64,
        type=str,
        required=False,
        help="When splitting up a long document into chunks, how much stride to "
             "take between chunks."
    )

    parser.add_argument(
        "--do_train",
        default=False,
        type=bool,
        required=False,
        help="Whether to run training."
    )

    parser.add_argument(
        "--do_predict",
        default=False,
        type=bool,
        required=False,
        help="Whether to run prediction."
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        required=False,
        help="Whether to run prediction."
    )

    parser.add_argument(
        "--predict_batch_size",
        default=36,
        type=int,
        required=False,
        help="Total batch size for predictions."
    )

    parser.add_argument(
        "--predict_file_shard_size",
        default=1000,
        type=int,
        required=False,
        help="The maximum number of examples to put into each temporary TF example file "
             "used as model input a prediction time."
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        required=False,
        help="The initial learning rate for Adam."
    )

    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        required=False,
        help="The initial learning rate for Adam."
    )

    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        required=False,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10% of training."
    )

    parser.add_argument(
        "--save_checkpoints_steps",
        default=0.1,
        type=float,
        required=False,
        help="How often to save the model checkpoint."
             "E.g., 0.1 = 10% of training."
    )

    parser.add_argument(
        "--iterations_per_loop",
        default=1000,
        type=int,
        required=False,
        help="How many steps to make in each estimator call."
    )

    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        required=False,
        help="The maximum length of an answer that can be generated. This is needed "
             "because the start and end predictions are not conditioned on one another."
    )

    parser.add_argument(
        "--include_unknowns",
        default=-1.0,
        type=float,
        required=False,
        help="If positive, probability of including answers of type `UNKNOWN`."
    )

    parser.add_argument(
        "--verbose_logging",
        default=False,
        type=bool,
        required=False,
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal TyDi evaluation."
    )

    parser.add_argument(
        "--max_passages",
        default=45,
        type=int,
        required=False,
        help="Maximum number of passages to consider for a "
             "single article. If an article contains more than"
             "this, they will be discarded during training. "
             "BERT's WordPiece vocabulary must be modified to include "
             "these within the [unused*] vocab IDs."
    )

    parser.add_argument(
        "--max_position",
        default=45,
        type=int,
        required=False,
        help="Maximum passage position for which to generate special tokens."
    )

    parser.add_argument(
        "--fail_on_invalid",
        default=True,
        type=bool,
        required=False,
        help="Stop immediately on encountering an invalid example? "
             "If false, just print a warning and skip it."
    )

    parser.add_argument(
        "--pretrained_weights",
        default=None,
        type=str,
        required=False,
        help="Pretrained weights to use in model predictions"
        "Produced by model.save_pretrained()"
    )

    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        required=False,
        help="weight decaying rate for adam optimizer"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=50000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--grad_acc_steps", default=2, type=int, help="Gradient accumulation steps")
    args = parser.parse_args()
    '''if args.do_train:
        model = TYDIQA.from_pretrained(args.init_checkpoint)
        model.to(DEVICE)
        train_dataset = torch.load(args.train_records_file)['dataset']
        train(args, train_dataset, model)
        logging.info("Finish Finetuning")'''
    if args.do_predict:
        model = TYDIQA.from_pretrained(args.pretrained_weights)
        model.to(DEVICE)
        predict(args, model)