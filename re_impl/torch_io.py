import collections
import gzip
import os

from torch.utils.data import Dataset
import torch
import data
import preprocess
import tokenization


# define dataset for tydi
class TyDiDataset(Dataset):
    """dataset_features should be a  list of dictionaries"""

    def __init__(self, dataset_features, is_training=True):
        self.features = dataset_features
        self.is_training = is_training

    def __getitem__(self, index):
        single_feature = self.features[index]
        unique_ids_tensors = torch.LongTensor(single_feature['unique_ids'])
        example_index_tensors = torch.LongTensor(single_feature['example_index'])
        input_ids_tensors = torch.LongTensor(single_feature['input_ids'])
        input_mask_tensors = torch.LongTensor(single_feature['input_mask'])
        segment_id_tensors = torch.LongTensor(single_feature['segment_ids'])
        language_id_tensors = torch.LongTensor(single_feature['language_id'])
        if self.is_training:
            start_position_tensors = torch.LongTensor(single_feature['start_positions'])
            end_position_tensors = torch.LongTensor(single_feature['end_positions'])
            answer_type_tensors = torch.LongTensor(single_feature['answer_types'])
            return {'unique_ids': unique_ids_tensors,
                    'example_index': example_index_tensors,
                    'input_ids': input_ids_tensors,
                    'input_mask': input_mask_tensors,
                    'segment_ids': segment_id_tensors,
                    'language_id': language_id_tensors,
                    'start_positions': start_position_tensors,
                    'end_positions': end_position_tensors,
                    'answer_types': answer_type_tensors}
        else:
            wp_start_offset_tensors = torch.LongTensor(single_feature['wp_start_offset'])
            wp_end_offset_tensors = torch.LongTensor(single_feature['wp_end_offset'])
            return {'unique_ids': unique_ids_tensors,
                    'example_index': example_index_tensors,
                    'input_ids': input_ids_tensors,
                    'input_mask': input_mask_tensors,
                    'segment_ids': segment_id_tensors,
                    'language_id': language_id_tensors,
                    'wp_start_offset': wp_start_offset_tensors,
                    'wp_end_offset': wp_end_offset_tensors}

    def __len__(self):
        return len(self.features)


class CreateTorchExampleFn(object):
    """Functor for creating TyDi tf.Examples to be written to a TFRecord file."""

    def __init__(self, is_training, max_question_length, max_seq_length,
                 doc_stride, include_unknowns, vocab_file):
        self.feature_lst = []
        self.is_training = is_training
        self.tokenizer = tokenization.TyDiTokenizer(vocab_file=vocab_file)
        self.max_question_length = max_question_length
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.include_unknowns = include_unknowns
        self.vocab = self.tokenizer.vocab  # used by callers

    def process(self, entry, errors, debug_info=None):
        """Converts TyDi entries into serialized tf examples.

    Args:
      entry: "TyDi entries", dicts as returned by `create_entry_from_json`.
      errors: A list that this function appends to if errors are created. A
        non-empty list indicates problems.
      debug_info: A dict of information that may be useful during debugging.
        These elements should be used for logging and debugging only. For
        example, we log how the text was tokenized into WordPieces.

    Yields:
      `tf.train.Example` with the features needed for training or inference
      (depending on how `is_training` was set in the constructor).
    """
        if not debug_info:
            debug_info = {}
        # convert raw data into TyDiExample
        tydi_example = data.to_tydi_example(entry, self.is_training)
        debug_info["tydi_example"] = tydi_example

        # Converts `TyDiExample`s into `InputFeatures` and sends them to `output_fn
        input_features = preprocess.convert_single_example(
            tydi_example,
            tokenizer=self.tokenizer,
            is_training=self.is_training,
            max_question_length=self.max_question_length,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            include_unknowns=self.include_unknowns,
            errors=errors,
            debug_info=debug_info)
        # convert example to features
        for input_feature in input_features:
            input_feature.example_index = int(entry["id"])
            input_feature.unique_id = (
                    input_feature.example_index + input_feature.doc_span_index)

            features = collections.OrderedDict()
            features["unique_ids"] = [input_feature.unique_id]
            features["example_index"] = [input_feature.example_index]
            features["input_ids"] = input_feature.input_ids
            features["input_mask"] = input_feature.input_mask
            features["segment_ids"] = input_feature.segment_ids
            features["language_id"] = [input_feature.language_id]

            if self.is_training:
                features["start_positions"] = [input_feature.start_position]
                features["end_positions"] = [input_feature.end_position]
                features["answer_types"] = [input_feature.answer_type]
            else:
                features["wp_start_offset"] = input_feature.wp_start_offset
                features["wp_end_offset"] = input_feature.wp_end_offset

            self.feature_lst.append(features)

        return self.feature_lst

    # this function convert feature_lst to dataset and return
    def convert_feature_to_dataset(self):
        self.dataset = TyDiDataset(self.feature_lst, is_training=self.is_training)

        return self.dataset

    def write_feature_to_file(self, data_dir):
        input_dir = data_dir if data_dir else "."
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}".format(
                "dev" if not self.is_training else "train"
            ),
        )
        torch.save({"features": self.feature_lst, "dataset": self.dataset}, cached_features_file)
