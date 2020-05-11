# Converts an TyDi dataset file to PyTorch Dataloader

# @author Zijing Gu

import argparse
import logging
import glob
import gzip
import json
import torch_io
import preprocess
import collections
import os
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser(description='Arguments for prepare tydi data.')

parser.add_argument('--input_jsonl',
                    default='tydiqa-v1.0-dev.jsonl.gz',
                    type=str,
                    metavar='PATH',
                    help='Path to Gzipped files containing NQ examples in Json format, one per line.')
# ??

parser.add_argument('--output_torch_data',
                    default='preprocessed_data/',
                    type=str,
                    metavar='PATH',
                    help='Output folder tydiData with all features extracted.')

parser.add_argument('--vocab_file',
                    default='mbert_modified_vocab.txt',
                    type=str,
                    metavar='PATH',
                    help='"The vocabulary file that the BERT model was trained on.')

parser.add_argument('--mql', '--max_question_length',
                    default=64,
                    type=int,
                    metavar='N',
                    help='The maximum number of tokens for the question. ')

parser.add_argument('--msl', '--max_seq_length',
                    default=512,
                    type=int,
                    metavar='N',
                    help='The maximum total input sequence length after WordPiece tokenization.')

parser.add_argument('--doc_stride',
                    default=128,
                    type=int,
                    metavar='N',
                    help='When splitting up a long doc into chunks, how much stride to take between chunks')

parser.add_argument('-e', '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

"""
Reads TyDi QA examples from JSONL files.
Args:
    input_jsonl_pattern: Glob of the gzipped JSONL files to read.
Yields:
      tydi_entry: "TyDiEntry"s, dicts as returned by `create_entry_from_json`,
        one per line of the input JSONL files.
"""


def read_entries(input_jsonl):
    for input_path in glob.glob(input_jsonl):
        with gzip.GzipFile(input_path) as in_file:
            # s = in_file.read()
            for line_num, line in enumerate(in_file):
                json_obj = json.loads(line, object_pairs_hook=collections.OrderedDict)
                entry = preprocess.create_entry_from_json(json_obj)

                if not entry:
                    logging.info("Invalid Example %d", json_obj["example_id"])

                yield entry


def main():
    args = parser.parse_args()
    if not os.path.exists(args.output_torch_data):
        print("creating new directory for output:", args.output_torch_data)
        os.mkdir(args.output_torch_data)

    data_processed = 0

    creator_fn = torch_io.CreateTorchExampleFn(
        is_training=not args.evaluate,
        max_question_length=args.mql,
        max_seq_length=args.msl,
        doc_stride=args.doc_stride,
        include_unknowns=-1,
        vocab_file=args.vocab_file
    )

    logging.info("Reading examples from glob: %s", args.input_jsonl)

    for entry in read_entries(args.input_jsonl):
        errors = []  # ?
        creator_fn.process(entry, errors)

        data_processed += 1
        # if data_processed >= 100:
        #     break
        if data_processed % 10 == 0:
            logging.info("Examples processed: %d", data_processed)

    creator_fn.convert_feature_to_dataset()
    creator_fn.write_feature_to_file(args.output_torch_data)


if __name__ == '__main__':
    main()
