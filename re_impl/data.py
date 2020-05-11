import collections

TextSpan = collections.namedtuple("TextSpan", "byte_positions text")

AnswerType = {'UNKNOWN': 0, 'YES': 1, 'NO': 2, 'MINIMAL': 3, 'PASSAGE': 4}

language = {'ARABIC': 0, 'BENGALI': 1, 'FINNISH': 2,
            'INDONESIAN': 3, 'JAPANESE': 4, 'SWAHILI': 5,
            'KOREAN': 6, 'RUSSIAN': 7, 'TELUGU': 8,
            'THAI': 9, 'ENGLISH': 10}


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
    """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

    def __new__(cls, type_, text=None, offset=None):
        return super(Answer, cls).__new__(cls, type_, text, offset)


class TyDiExample(object):
    """A single training/test example.

  Typically created by `to_tydi_example`. This class is a fairly straightforward
  serialization of the dict-based entry format created in
  `create_entry_from_json`.
  """

    def __init__(self,
                 example_id,
                 language_id,
                 question,
                 contexts,
                 plaintext,
                 context_to_plaintext_offset,
                 answer=None,
                 start_byte_offset=None,
                 end_byte_offset=None):
        self.example_id = example_id
        self.language_id = language_id
        self.question = question
        self.contexts = contexts
        self.plaintext = plaintext
        self.context_to_plaintext_offset = context_to_plaintext_offset
        self.answer = answer  # type: Answer
        self.start_byte_offset = start_byte_offset
        self.end_byte_offset = end_byte_offset


def get_first_annotation(json_dict, max_passages):
    if "annotations" not in json_dict:
        return None, -1, (-1, -1)

    positive_annotations = sorted(
        [a for a in json_dict["annotations"] if a["passage_answer"]["candidate_index"] >= 0],
        key=lambda a: a["passage_answer"]["candidate_index"])

    for a in positive_annotations:
        if a["minimal_answer"]:
            # Check if it is a non null answer.
            start_byte_offset = a["minimal_answer"]["plaintext_start_byte"]
            if start_byte_offset < 0:
                continue

            idx = a["passage_answer"]["candidate_index"]
            if idx >= max_passages:
                continue
            end_byte_offset = a["minimal_answer"]["plaintext_end_byte"]

            annotated_start = start_byte_offset - json_dict["passage_answer_candidates"][idx]["plaintext_start_byte"]
            annotated_end = end_byte_offset - json_dict["passage_answer_candidates"][idx]["plaintext_start_byte"]

            return a, idx, (annotated_start, annotated_end)

    for a in positive_annotations:
        idx = a["passage_answer"]["candidate_index"]
        if idx >= max_passages:
            continue
        return a, idx, (-1, -1)

    return None, -1, (-1, -1)


def byte_str(text):
    return text.encode("utf-8")


def byte_len(text):
    return len(byte_str(text))


def byte_slice(text, start, end, errors="replace"):
    return byte_str(text)[start:end].decode("utf-8", errors=errors)


def get_text_span(example, span):
    """Returns the text in the example's document in the given span."""
    byte_positions = []
    # `text` is a byte string since `document_plaintext` is also a byte string.
    start = span["plaintext_start_byte"]
    end = span["plaintext_end_byte"]
    text = byte_slice(example["document_plaintext"], start, end)
    for i in range(start, end):
        byte_positions.append(i)
    return TextSpan(byte_positions, text)


def get_candidate_text(json_dict, idx):
    """Returns a text representation of the candidate at the given index."""
    # No candidate at this index.
    if idx < 0 or idx >= len(json_dict["passage_answer_candidates"]):
        raise ValueError("Invalid index for passage candidate: {}".format(idx))

    return get_text_span(json_dict, json_dict["passage_answer_candidates"][idx])


def candidates_iter(json_dict):
    """Yields the candidates that should not be skipped in an example."""
    for idx, cand in enumerate(json_dict["passage_answer_candidates"]):
        yield idx, cand


def make_tydi_answer(contexts, answer):
    start = answer["start"]
    end = answer["end"]
    input_text = answer["type"]

    if (answer["candidate_id"] == -1 or start >= byte_len(contexts) or
            end > byte_len(contexts)):
        answer_type = AnswerType['UNKNOWN']
        start, end = 0, 1
    else:
        answer_type = AnswerType[input_text.upper()]
    return Answer(
        answer_type, text=byte_slice(contexts, start, end), offset=start)


def get_language_id(input_text):
    if input_text.upper() in language:
        language_id = language[input_text.upper()]
    else:
        raise ValueError("Invalid language <%s>" % input_text)
    return language_id


def to_tydi_example(entry, is_training):
    """Converts a TyDi 'entry' from `create_entry_from_json` to `TyDiExample`."""

    if is_training:
        answer = make_tydi_answer(entry["contexts"], entry["answer"])
        start_byte_offset = answer.offset
        end_byte_offset = answer.offset + byte_len(answer.text)
    else:
        answer = None
        start_byte_offset = None
        end_byte_offset = None

    return TyDiExample(
        example_id=int(entry["id"]),
        language_id=get_language_id(entry["language"]),
        question=entry["question"],
        contexts=entry["contexts"],
        plaintext=entry["plaintext"],
        context_to_plaintext_offset=entry["context_to_plaintext_offset"],
        answer=answer,
        start_byte_offset=start_byte_offset,
        end_byte_offset=end_byte_offset)
