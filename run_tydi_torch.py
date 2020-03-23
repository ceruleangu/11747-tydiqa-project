import collections
import json
import os

from absl import logging
from bert import modeling as bert_modeling
import tensorflow.compat.v1 as tf
import postproc
import preprocess
import torch_io
import tydi_modeling

import torch
import argparse
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, squad_convert_examples_to_features
from tydi_modeling_torch import TYDIQA
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import tqdm
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Arguments for running tydi')
parser.add_argument()

parser.add_argument(
    "bert_config_file", default=None,
    help="The config json file corresponding to the pre-trained BERT model. "
         "This specifies the model architecture.")

parser.add_argument(
    "vocab_file", default=None,
    help="The vocabulary file that the BERT model was trained on.")

parser.add_argument(
    "output_dir", default=None,
    help="The output directory where the model checkpoints will be written.")

parser.add_argument(
    "train_records_file", default=None,
    help="Precomputed tf records for training.")

parser.add_argument(
    "record_count_file", default=None,
    help="File containing number of precomputed training records "
         "(in terms of 'features', meaning slices of articles). "
         "This is used for computing how many steps to take in "
         "each fine tuning epoch.")

parser.add_argument(
    "candidate_beam", default=30,
    help="How many wordpiece offset to be considered as boundary at inference time.")

parser.add_argument(
    "predict_file", default=None,
    help="TyDi json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz. "
         "Used only for `--do_predict`.")

parser.add_argument(
    "precomputed_predict_file", default=None,
    help="TyDi tf.Example records for predictions, created separately by "
         "`prepare_tydi_data.py` Used only for `--do_predict`.")

parser.add_argument(
    "output_prediction_file", default=None,
    help="Where to print predictions in TyDi prediction format, to be passed to"
         "tydi_eval.py.")

parser.add_argument(
    "init_checkpoint", default=None,
    help="Initial checkpoint (usually from a pre-trained mBERT model).")

parser.add_argument(
    "max_seq_length", default=512,
    help="The maximum total input sequence length after WordPiece tokenization. "
         "Sequences longer than this will be truncated, and sequences shorter "
         "than this will be padded.")

parser.add_argument(
    "doc_stride", default=128,
    help="When splitting up a long document into chunks, how much stride to "
         "take between chunks.")

parser.add_argument(
    "max_question_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

parser.add_argument("do_train", False, "Whether to run training.")

parser.add_argument("do_predict", False, "Whether to run prediction.")

parser.add_argument("train_batch_size", 16, "Total batch size for training.")

parser.add_argument("predict_batch_size", 8,
                    "Total batch size for predictions.")

parser.add_argument(
    "predict_file_shard_size", 1000,
    "The maximum number of examples to put into each temporary TF example file "
    "used as model input a prediction time.")

parser.add_argument("learning_rate", 5e-5, "The initial learning rate for Adam.")

parser.add_argument("num_train_epochs", 3.0,
                    "Total number of training epochs to perform.")

parser.add_argument(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

parser.add_argument("save_checkpoints_steps", 1000,
                    "How often to save the model checkpoint.")

parser.add_argument("iterations_per_loop", 1000,
                    "How many steps to make in each estimator call.")

parser.add_argument(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

parser.add_argument(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

parser.add_argument(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal TyDi evaluation.")

parser.add_argument(
    "max_passages", 45, "Maximum number of passages to consider for a "
                        "single article. If an article contains more than"
                        "this, they will be discarded during training. "
                        "BERT's WordPiece vocabulary must be modified to include "
                        "these within the [unused*] vocab IDs.")

parser.add_argument(
    "max_position", 45,
    "Maximum passage position for which to generate special tokens.")

parser.add_argument(
    "fail_on_invalid", True,
    "Stop immediately on encountering an invalid example? "
    "If false, just print a warning and skip it.")

parser.add_argument(
    "adam_epsilon", default=1e-8,
    help="weight decaying rate for adam optimizer"
)

parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)

parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")

args = parser.parse_args()


def train(args, train_dataset, dev_dataset, model, tokenizer):
    # use tensorboard to keep track of training process
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)

    logging.info("  Let's start finetuning!")
    tr_loss, logging_loss = 0.0, 0.0
    global_step = 0
    for epoch in tqdm.trange(args.num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch.to(DEVICE)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
                answer_types=batch["answer_types"]
            )

            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            # loggin points
            if global_step % args.logging_steps == 0:

                # todo: evaluate does not have returns
                results = evaluate(args, dev_dataset, model, tokenizer)
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            # save checkpoint
            if global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                logging.info("Saving optimizer and scheduler states to %s", output_dir)

    return global_step, tr_loss / global_step


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logging.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logging.info("Creating features from dataset file at %s", input_dir)

        # Replace this with processor we defined for this specific task
        # Look at source code here at: https://github.com/huggingface/transformers/blob/cf72
        # 479bf11bf7fbc499a518896dfd3cafdd0b21/src/transformers/data/processors/squad.py#L566

        processor = SquadV2Processor()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        logging.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def evaluate(args, model, tokenizer):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)

    # Eval!
    logging.info("***** Running evaluation*****")
    logging.info("  Num examples = %d", len(dataset))
    logging.info("  Batch size = %d", args.predict_batch_size)

    all_results = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch.to(DEVICE)

        with torch.no_grad():

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids']
            )
            # todo ??
            all_results.append(outputs)
            unique_ids = batch["unique_ids"]

        # precomputed_predict_file is produced and cached by preprocess
        if not args.precomputed_predict_file:
            predict_examples_iter = preprocess.read_tydi_examples(
                input_file=args.predict_file,
                is_training=False,
                max_passages=args.max_passages,
                max_position=args.max_position,
                fail_on_invalid=args.fail_on_invalid,
                # open_fn=torch_io.gopen
            )

    return all_results


def predict():
    pass


def main():
    logging.set_verbosity(logging.INFO)
    bert_config = BertConfig.from_json_file(args.bert_config_file)


if __name__ == "__main__":
    main()
