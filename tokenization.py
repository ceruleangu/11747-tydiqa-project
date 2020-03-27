
import collections
from absl import logging
import bert_tokenization
import data
import re

SubToken = collections.namedtuple("SubToken",["normalized","orig","is_good"])


def split_token(subtokens, should_isolate_func, are_good):
  output = []
  result_subtoken = ''
  for subtoken, orig_subtoken, is_good in subtokens:
    assert subtoken == orig_subtoken
    if not is_good:
      output.append(SubToken(subtoken, subtoken, is_good=False))
      continue

    for char in subtoken:
      if should_isolate_func(char):
        if result_subtoken:
          output.append(
              SubToken(result_subtoken, result_subtoken, is_good=True))
          result_subtoken = ''
        output.append(SubToken(char, char, is_good=are_good))
      else:
        result_subtoken+=char
  if result_subtoken:
    output.append(SubToken(result_subtoken, result_subtoken, is_good=True))
  return output

def white_tokenize(subtokens):
  return split_token(subtokens, lambda char: char.isspace(), are_good=True)

class FullTokenizer(object):


  def __init__(self, vocab_file):
    self.vocab = bert_tokenization.load_vocab(vocab_file)
    self.inv_vocab = {value: key for key, value in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(vocab=self.vocab)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):

    split_tokens = []
    for token, orig_token, is_good_token in self.basic_tokenizer.tokenize(text):
      if not is_good_token:
        split_tokens.append(SubToken(token, orig_token, is_good=False))
        continue

      if bert_tokenization.preserve_token(token, self.vocab):
        split_tokens.append(SubToken(token, orig_token, is_good=True))
        continue

      for sub_token in self.wordpiece_tokenizer.tokenize([SubToken(token, orig_token, is_good_token)]):
        split_tokens.append(sub_token)

    return split_tokens


class BasicTokenizer(bert_tokenization.BasicTokenizer):


  def __init__(self, vocab=tuple()):

    self.vocab = vocab

  def tokenize(self, text):

    text = bert_tokenization.convert_to_unicode(text)


    subtokens = [SubToken(text, text, is_good=True)]
    subtokens = white_tokenize(self._tokenize_chinese_chars(self._clean_text(subtokens)))


    split_subtokens = []
    for subtoken, orig_subtoken, is_good in subtokens:

      if not is_good:
        split_subtokens.append(SubToken(subtoken, subtoken, is_good=False))
        continue

      if bert_tokenization.preserve_token(subtoken, self.vocab):
        split_subtokens.append(SubToken(subtoken, subtoken, is_good=True))
        continue

      split_subtokens.extend(
          self._run_split_on_punc([SubToken(subtoken, subtoken, is_good=True)]))
    return split_subtokens

  def _run_split_on_punc(self, subtokens):
    return split_token(subtokens, lambda char: bert_tokenization._is_control(self,char), are_good=True)


  def _is_chinese_char(self, cp):
    return bert_tokenization.BasicTokenizer._is_chinese_char(  self, cp)

  def _tokenize_chinese_chars(self, subtokens):
    return split_token(
        subtokens, lambda char: self._is_chinese_char(ord(char)), are_good=True)

  def _clean_text(self, subtokens):

    def is_isolate(char):
      return self._is_control(char) or ord(char) == 0 or ord(char) == 0xfffd

    return split_token(subtokens, is_isolate, are_good=False)


class WordpieceTokenizer(object):


  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, subtokens):
    output_tokens = []
    for token, orig_token, is_good in subtokens:
      if not is_good:
        output_tokens.append(SubToken(token, orig_token, is_good=False))
        continue
      token_char_len = len(token)
      if token_char_len > self.max_input_chars_per_word:
        output_tokens.append(SubToken(self.unk_token, token, is_good=True))
        continue
      sub_tokens = []
      is_unk = False
      start = 0
      while start < token_char_len:
        cur_substr = None
        end = token_char_len
        orig_substr=''
        while start < end:
          orig_substr = token[start:end]
          substr = orig_substr
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr:
          sub_tokens.append(SubToken(cur_substr, orig_substr, is_good=True))
        else:
          is_unk = True
          break
        start = end
      if not is_unk:
        output_tokens.extend(sub_tokens)
      else:
        output_tokens.append(SubToken(self.unk_token, token, is_good=True))

    return output_tokens


class TyDiToken(object):


  def __init__(self, vocab_file, fail_on_mismatch=False):
    self.vocab = bert_tokenization.load_vocab(vocab_file)
    self.tokenizer = FullTokenizer(vocab_file=vocab_file)
    self.fail_on_mismatch = fail_on_mismatch
    self.unk_token = "[UNK]"

  def tokenize(self, text):
    whitespace_tokens = text.split(" ")
    unk_id = self.vocab[self.unk_token]
    wordpieces = []
    starts = []
    limits = []
    mismatched_tokens = []
    mismatch_bytes = 0
    num_tokens = len(whitespace_tokens)

    for token in whitespace_tokens:
      internal_byte_offset = 0
      subtokens = self.tokenizer.tokenize(token)
      subtoken_ids_lengths = [ (self.vocab.get(subtoken, unk_id),data.byte_len(orig_subtoken))
                               for subtoken, orig_subtoken, _ in subtokens]


      actual_token_length = data.byte_len(token)
      actual_subtokens_length = sum(subtoken_lengths for _,subtoken_lengths in subtoken_ids_lengths )

      if actual_token_length != actual_subtokens_length:
        mismatch_bytes += abs(actual_token_length - actual_subtokens_length)
        mismatched_tokens.append(token)

        if self.fail_on_mismatch:
          raise ValueError("Mismatched token")

      inside_berttok_wordpieces = [subtoken_id for subtoken_id,_ in subtoken_ids_lengths]
      inside_berttok_starts = []
      inside_berttok_limits = []
      for subtoken_id, subtoken_len in subtoken_ids_lengths:
        inside_berttok_starts.append(internal_byte_offset)
        inside_berttok_limits.append(internal_byte_offset + subtoken_len)
        internal_byte_offset += subtoken_len
      wordpieces.append(inside_berttok_wordpieces)
      starts.append(inside_berttok_starts)
      limits.append(inside_berttok_limits)
    mismatch = " ".join(mismatched_tokens)
    if mismatched_tokens:
      logging.info("Have {} mismatched tokens of %d ({} bytes off): {}".format(
                   len(mismatched_tokens), num_tokens, mismatch_bytes,mismatch))

    out,start_offsets_out,end_offsets_out  = [],[],[]
    offset_to_wp_out = {}
    curr_offset,count = 0,0

    for token, wps, wp_starts, wp_limits in zip(whitespace_tokens, wordpieces,
                                                starts, limits):

      if self.is_special_token(token):
        if token in self.vocab:
          vocab_id = self.vocab[token]
          for j in range(data.byte_len(token)):
            offset_to_wp_out[j + curr_offset] = len(out)
          if vocab_id > -1:
            out.append(vocab_id)
          else:
            vocab_id = self.vocab["[UNK]"]
            out.append(vocab_id)
          start_offsets_out.append(curr_offset)
          out.append(curr_offset + data.byte_len(token) - 1)
      else:
        for i, wp in enumerate(zip(wp_starts, wp_limits)):
          wp_start,wp_limit = wp[0],wp[1]
          for j in range(data.byte_len(token)):
            if j < wp_start or j >= wp_limit:
              continue
            else:
              offset_to_wp_out[j + curr_offset] = len(out) + i
        out.extend(wps)
        start_offsets_out.extend([k + curr_offset for k in wp_starts])
        end_offsets_out.extend([k + curr_offset - 1 for k in wp_limits])
      curr_offset += data.byte_len(token)

      count += 1
      if count < len(whitespace_tokens):
        offset_to_wp_out[curr_offset] = -1
        curr_offset += 1
    return out

  def _flatten_inner(self, seq):
    result = []
    for subseq in seq:
      inner = []
      for subsubseq in subseq:
        inner.extend(subsubseq)
      result+=[inner]
    return result

  def is_special_token(self, token):

    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[Q]", "[YES]", "[NO]", "[NoLongAnswer]", "[NoShortAnswer]", "[SA]",
                      "[/SA]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    if token in special_tokens:
      return True
    elif re.match(r'^\[Paragraph=',token) or re.match(r'^\[ContextId=',token):
      return True
    else:
      return False



