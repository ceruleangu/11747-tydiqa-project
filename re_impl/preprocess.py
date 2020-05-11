"""Performs model-specific preprocessing.

This includes tokenization and adding special tokens to the input.

This module does not have any dependencies on TensorFlow and should be re-usable
within your favorite ML/DL framework.
"""

import collections
import functools
import glob
import json
import random

import logging
import data
import tokenization

"""Creates an TyDi 'entry' from the raw JSON.

Args:
    json_dict: A single JSONL line, deserialized into a dict.
    max_passages: Maximum number of passages to consider for a single article.
    max_position: Maximum passage position for which to generate special tokens.
    
Returns:
    'TyDiEntry' type: a dict-based format consumed by downstream functions:
    entry = {
        "name": str,
        "id": str,
        "language": str,
        "question": {"input_text": str},
        "answer": {
          "candidate_id": annotated_idx,
          "span_text": "",
          "span_start": -1,
          "span_end": -1,
          "input_text": "passage",
        }
        "has_correct_context": bool,
        # Includes special tokens appended.
        "contexts": str,
        # Context index to byte offset in `contexts`.
        "context_to_plaintext_offset": Dict[int, int],
        "plaintext" = json_dict["document_plaintext"]
    }
"""


def create_entry_from_json(json_obj, max_passages=45, max_position=45):
    entry = {'document_title': json_obj['document_title'],
             'id': json_obj['example_id'],
             'language': json_obj['language'],
             'question': json_obj['question_text']}

    annotation, candidate_idx, annotated_start_end = data.get_first_annotation(
        json_obj, max_passages)
    answer = {'candidate_id': candidate_idx,
              'type': 'passage',
              'span': '',
              'start': -1,
              'end': -1
              }
    # if annotated
    if annotation is not None:
        # if Yes/no answers, added in type.
        if annotation['yes_no_answer'] != 'NONE':
            answer['type'] = annotation['yes_no_answer'].lower()
        # if has minimal answer span
        if annotated_start_end != (-1, -1):
            answer['type'] = 'minimal'
            start = annotated_start_end[0]
            end = annotated_start_end[1]
            text = data.get_candidate_text(json_obj, candidate_idx).text
            answer['span'] = data.byte_slice(text, start, end)
            answer['start'] = start
            answer['end'] = end
        # passage selected
        if annotation['passage_answer']['candidate_index'] >= 0:
            answer['span'] = data.get_candidate_text(json_obj, candidate_idx).text
            answer['start'] = 0
            answer['end'] = data.byte_len(answer['span'])
    entry['answer'] = answer

    paragraph_idx = []
    paragraph_context = []

    # add candidate paragraph types and positions
    # ct = 0
    # for _, candidate in data.candidates_iter(json_obj):
    #     if ct < max_position:
    #         ct += 1
    #         candidate["type_and_position"] = "[Paragraph=%d]" % ct
    #     else: break

    for idx, _ in data.candidates_iter(json_obj):
        res = data.get_candidate_text(json_obj, idx)
        context = {"id": idx,
                   # "type": "[NoLongAnswer]" if idx == -1 else json_obj["passage_answer_candidates"][idx]["type_and_position"],
                   "text_range": res[0],
                   "text": res[1]}
        # Get list of all byte positions of the candidate and its plaintext.
        # Unpack `TextSpan` tuple.
        paragraph_idx.append(idx)
        paragraph_context.append(context)
        if len(paragraph_idx) >= max_passages:
            break
    # entry['has_correct_context'] = candidate_idx in paragraph_idx

    all_contexts_with_tokens = []
    offset = 0  # a byte offset relative to `contexts` (concatenated candidate passages with special tokens added).
    context_to_plaintext_offset = []
    for idx, context in zip(paragraph_idx, paragraph_context):
        special_token = "[ContextId={}]".format(idx)
        all_contexts_with_tokens.append(special_token)

        context_to_plaintext_offset.append([-1] * data.byte_len(special_token))
        # Account for the special token and its trailing space (due to the join
        # operation below)
        offset += data.byte_len(special_token) + 1

        if context["id"] == candidate_idx:
            answer["start"] += offset
            answer["end"] += offset
        if context["text"]:
            all_contexts_with_tokens.append(context["text"])
            # Account for the text and its trailing space (due to the join operation below)
            offset += data.byte_len(context["text"]) + 1
            context_to_plaintext_offset.append(context["text_range"])

        # When we join the contexts together with spaces below, we'll add an extra
        # byte to each one, so we have to account for these by adding a -1 (no
        # assigned wordpiece) index at each *boundary*. It's easier to do this here
        # than above since we don't want to accidentally add extra indices after the
        # last context.
    context_to_plaintext_offset = functools.reduce(
        lambda a, b: a + [-1] + b, context_to_plaintext_offset)

    entry["contexts"] = " ".join(all_contexts_with_tokens)
    entry["context_to_plaintext_offset"] = context_to_plaintext_offset
    entry["plaintext"] = json_obj["document_plaintext"]

    return entry


def find_nearest_wordpiece_index(offset_index, offset_to_wp, scan_right):
    """According to offset_to_wp dictionary, find the wordpiece index for offset.

  Some offsets do not have mapping to word piece index if they are delimited.
  If scan_right is True, we return the word piece index of nearest right byte,
  nearest left byte otherwise.

  Args:
    offset_index: the target byte offset.
    offset_to_wp: a dictionary mapping from byte offset to wordpiece index.
    scan_right: When there is no valid wordpiece for the offset_index, will
      consider offset_index+i if this is set to True, offset_index-i otherwise.

  Returns:
    The index of the nearest word piece of `offset_index`
    or -1 if no match is possible.
  """

    for i in range(0, len(offset_to_wp.items())):
        if scan_right:
            next_ind = offset_index + i
        else:
            next_ind = offset_index - i

        if next_ind >= 0 and next_ind in offset_to_wp:
            return_ind = offset_to_wp[next_ind]
            # offset has a match.
            if return_ind > -1:
                return return_ind
    return -1


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 language_id,
                 doc_span_index,
                 wp_start_offset,
                 wp_end_offset,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 answer_text="",
                 answer_type=data.AnswerType['MINIMAL']):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.language_id = language_id
        self.wp_start_offset = wp_start_offset
        self.wp_end_offset = wp_end_offset
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position  # this is a wordpiece index.
        self.end_position = end_position
        self.answer_text = answer_text
        self.answer_type = answer_type


"""Converts a single `TyDiExample` into a list of InputFeatures.

Args:
    tydi_example: `TyDiExample` from a single JSON line in the corpus.
    tokenizer: Tokenizer object that supports `tokenize` and
      `tokenize_with_offsets`.
    is_training: Are we generating these examples for training? (as opposed to
      inference).
    max_question_length: see FLAGS.max_question_length.
    max_seq_length: see FLAGS.max_seq_length.
    doc_stride: see FLAGS.doc_stride.
    include_unknowns: see FLAGS.include_unknowns.
    errors: List to be populated with error strings.
    debug_info: Dict to be populated with debugging information (e.g. how the
      strings were tokenized, etc.)

  Returns:
    List of `InputFeature`s.
"""


def convert_single_example(
        tydi_example,
        tokenizer,
        is_training,
        max_question_length,
        max_seq_length,
        doc_stride,
        include_unknowns,
        errors,
        debug_info=None):
    features = []
    question = "[Q] " + tydi_example.question
    question_wordpieces = tokenizer.tokenize(question)
    if len(question_wordpieces) > max_question_length:
        question_wordpieces = question_wordpieces[-max_question_length:]

    # `tydi_example.contexts` includes the entire document (article) worth of
    # candidate passages concatenated with special tokens such as '[ContextId=0]'.
    (contexts_wordpieces, contexts_start_offsets, contexts_end_offsets,
     contexts_offset_to_wp) = (tokenizer.tokenize_with_offsets(tydi_example.contexts))

    wp_start_offsets = [tydi_example.context_to_plaintext_offset[i] if i >= 0 else -1 for i in contexts_start_offsets]
    wp_end_offsets = [tydi_example.context_to_plaintext_offset[i] if i >= 0 else -1 for i in contexts_end_offsets]

    # The -3 accounts for
    # 1. [CLS] -- Special BERT class token, which is always first.
    # 2. [SEP] -- Special separator token, placed after question.
    # 3. [SEP] -- Special separator token, placed after article content.
    max_wordpieces_for_doc = max_seq_length - len(question_wordpieces) - 3

    doc_spans = split_doc_spans(contexts_wordpieces, doc_stride, max_wordpieces_for_doc)

    if is_training:
        wp_end_position, wp_start_position = find_doc_wp_range(contexts_offset_to_wp, tydi_example)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        segment_ids, wp_end_offset, wp_start_offset, wps = generate_doc_span_wordpiece(contexts_wordpieces, doc_span,
                                                                                       question_wordpieces, tokenizer,
                                                                                       wp_end_offsets, wp_start_offsets)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(wps)
        pad_sequence(input_mask, max_seq_length, segment_ids, wp_end_offset, wp_start_offset, wps)

        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (
                    wp_start_position >= doc_span.start and wp_end_position <= doc_span.start + doc_span.length - 1)
            if ((not contains_an_annotation) or
                    tydi_example.answer.type == data.AnswerType['UNKNOWN']):
                # If an example has unknown answer type or does not contain the answer
                # span, then ignore it.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.
                continue
            else:
                answer_text, answer_type, end_position, start_position = get_doc_span_answer(doc_span,
                                                                                             question_wordpieces,
                                                                                             tydi_example,
                                                                                             wp_end_offset,
                                                                                             wp_end_position,
                                                                                             wp_start_offset,
                                                                                             wp_start_position)
        else:
            start_position = None
            end_position = None
            answer_type = None
            answer_text = ""

        feature = InputFeatures(
            unique_id=-1,  # this gets assigned afterwards.
            example_index=tydi_example.example_id,
            language_id=tydi_example.language_id,
            doc_span_index=doc_span_index,
            wp_start_offset=wp_start_offset,
            wp_end_offset=wp_end_offset,
            input_ids=wps,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            answer_text=answer_text,
            answer_type=answer_type)

        features.append(feature)

    return features


def get_doc_span_answer(doc_span, question_wordpieces, tydi_example, wp_end_offset, wp_end_position, wp_start_offset,
                        wp_start_position):
    doc_offset = len(question_wordpieces) + 2  # one for CLS, one for SEP.
    start_position = wp_start_position - doc_span.start + doc_offset
    end_position = wp_end_position - doc_span.start + doc_offset
    answer_type = tydi_example.answer.type
    answer_start_byte_offset = wp_start_offset[start_position]
    answer_end_byte_offset = wp_end_offset[end_position]
    answer_text = tydi_example.contexts[
                  answer_start_byte_offset:answer_end_byte_offset]
    return answer_text, answer_type, end_position, start_position


def split_doc_spans(contexts_wordpieces, doc_stride, max_wordpieces_for_doc):
    doc_span = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    doc_span_start_wp_offset = 0
    len_contexts_wordpieces = len(contexts_wordpieces)
    while doc_span_start_wp_offset < len_contexts_wordpieces:
        length = len_contexts_wordpieces - doc_span_start_wp_offset
        length = min(length, max_wordpieces_for_doc)
        doc_spans.append(doc_span(start=doc_span_start_wp_offset, length=length))
        if doc_span_start_wp_offset + length == len_contexts_wordpieces:
            break
        doc_span_start_wp_offset += min(length, doc_stride)
    return doc_spans


def find_doc_wp_range(contexts_offset_to_wp, tydi_example):
    wp_start_position = find_nearest_wordpiece_index(
        tydi_example.start_byte_offset, contexts_offset_to_wp, True)
    wp_end_position = find_nearest_wordpiece_index(
        tydi_example.end_byte_offset - 1, contexts_offset_to_wp, False)
    return wp_end_position, wp_start_position


def pad_sequence(input_mask, max_seq_length, segment_ids, wp_end_offset, wp_start_offset, wps):
    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(wps))
    padding_offset = [-1] * (max_seq_length - len(wps))
    wps.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)
    wp_start_offset.extend(padding_offset)
    wp_end_offset.extend(padding_offset)


def generate_doc_span_wordpiece(contexts_wordpieces, doc_span, question_wordpieces, tokenizer, wp_end_offsets,
                                wp_start_offsets):
    wps = []
    wps.append(tokenizer.get_vocab_id("[CLS]"))
    segment_ids = []
    segment_ids.append(0)
    wps.extend(question_wordpieces)
    segment_ids.extend([0] * len(question_wordpieces))
    wps.append(tokenizer.get_vocab_id("[SEP]"))
    segment_ids.append(0)
    wp_start_offset = [-1] * len(wps)
    wp_end_offset = [-1] * len(wps)
    for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        wp_start_offset.append(wp_start_offsets[split_token_index])
        wp_end_offset.append(wp_end_offsets[split_token_index])
        wps.append(contexts_wordpieces[split_token_index])
        segment_ids.append(1)
    wps.append(tokenizer.get_vocab_id("[SEP]"))
    wp_start_offset.append(-1)
    wp_end_offset.append(-1)
    segment_ids.append(1)
    return segment_ids, wp_end_offset, wp_start_offset, wps


"""Read a TyDi json file into a list of `TyDiExample`.

  Delegates to `preproc.create_entry_from_json` to add special tokens to
  input and handle character offset tracking.

  Args:
    input_file: Path or glob to input JSONL files to be read (possibly gzipped).
    is_training: Should we create training samples? (as opposed to eval
      samples).
    max_passages: See FLAGS.max_passages.
    max_position: See FLAGS.max_position.
    fail_on_invalid: Should we immediately stop processing if an error is
      encountered?
    open_fn: A function that returns a file object given a path. Usually
      `tf_io.gopen`; could be standard Python `open` if using this module
      outside Tensorflow.

  Yields:
    `TyDiExample`s
"""


def read_tydi_examples(input_file,
                       is_training=False,
                       max_passages=45,
                       max_position=45):
    input_paths = glob.glob(input_file)

    n = 0
    for path in input_paths:
        logging.info("Reading: %s", path)
        with open(path) as input_file:
            logging.info(path)
            for line in input_file:
                json_dict = json.loads(line, object_pairs_hook=collections.OrderedDict)
                entry = create_entry_from_json(
                    json_dict,
                    max_passages=max_passages,
                    max_position=max_position
                    # fail_on_invalid=fail_on_invalid
                )
                if entry:
                    tydi_example = data.to_tydi_example(entry, is_training)
                    n += 1
                    yield tydi_example
