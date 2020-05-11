
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata


import six
import argparse



# parser = argparse.ArgumentParser()
# parser.add_argument('preserve_unused_tokens',default='False',action='store_true',\
#                     help='If True, Wordpiece tokenization will not be applied to words in the vocab.')
# args = parser.parse_args()

_UNUSED_TOKEN_RE = re.compile("^\\[unused\\d+\\]$")

preserve_unused_tokens = False

def preserve_token(token, vocab):

  if token not in vocab:
    return False
  if not preserve_unused_tokens:
    return False
  preserved_YN = False
  if _UNUSED_TOKEN_RE.search(token):
    preserved_YN = True
  return preserved_YN


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):

  if not init_checkpoint:
    return
  elif not re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint) :
    return
  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  model_name = m.group(1)

  lower_models = { "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
                   "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"}

  cased_models = {"cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
                   "multi_cased_L-12_H-768_A-12"}

  error_info = {}
  if model_name in lower_models and not do_lower_case:
    error_info['is_bad_config'] = True
    error_info['actual_flag'] = False
    error_info['case_name'] = "lowercased"
    error_info['opposite_flag'] = True
    raise ValueError(
      "model name :{} \
      init_checkpoint：{}\
      case_name:{}\
      actual_flag:{}".format(model_name,init_checkpoint, error_info['case_name'],str(error_info['actual_flag'])))

  elif model_name in cased_models and do_lower_case:
    error_info['is_bad_config'] = True
    error_info['actual_flag']= True
    error_info['case_name'] = "cased"
    error_info['opposite_flag'] = False
    raise ValueError(
      "model name :{} \
      init_checkpoint：{}\
      case_name:{}\
      actual_flag:{}".format(model_name,init_checkpoint, error_info['case_name'],str(error_info['actual_flag'])))





def convert_to_unicode(text):
  if six.PY2:
    if type(text)== str:
      text = text.decode("utf-8", "ignore")
      return text
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Error！ It is an unsupported string type: %s".format(type(text)))
  if six.PY3:
    if type(text)== str:
      return text
    elif type(text)== bytes:
      text = text.decode("utf-8", "ignore")
      return text
    else:
      raise ValueError("Error！ It is an unsupported string type: %s".format(type(text)))
  else:
    raise ValueError("Error! It is not running on Python2 or Python 3?")


def printable_text(text):
  if six.PY2:
    if type(text)== str:
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Error！ It is an unsupported string type: %s".format(type(text)))

  if six.PY3:
    if type(text)== str:
      return text
    elif type(text)== bytes:
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Error！ It is an unsupported string type: %s".format(type(text)))

  else:
    raise ValueError("Error! It is not running on Python2 or Python 3?")


def load_vocab(vocab_file):

  vocab = collections.defaultdict(int)
  with open(vocab_file, "r") as reader:
    for line in reader:
      token = convert_to_unicode(line).strip()
      vocab[token] = len(vocab)
  return vocab



def convert_by_vocab(vocab, items):
  return [vocab[item] for item in items]





class BasicTokenizer(object):

  def __init__(self, do_lower_case=True, vocab=tuple()):
    self.do_lower_case = do_lower_case
    self.vocab = vocab

  def tokenize(self, text):
    text = self._tokenize_chinese_chars(text)
    if not text:
      orig_tokens =[]
    else:
      orig_tokens=text.strip().split()
    split_tokens = []
    for token in orig_tokens:
      if preserve_token(token, self.vocab):
        split_tokens.append(token)
        continue
      if self.do_lower_case:
        token = self._run_strip_accents(token.lower())
      split_tokens+=self._run_split_on_punc(token)

    output_tokens = " ".join(split_tokens)
    if output_tokens:
      return output_tokens.strip().split()
    else:
      return []

  def _run_strip_accents(self, text):
    output = ""
    text = unicodedata.normalize("NFD", text)
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output += char
    return output

  def _run_split_on_punc(self, text):
    chars = list(text)
    start_new_word = True
    output = []
    for i in range(len(chars)):
      char = chars[i]
      if _is_punctuation(char):
        start_new_word = True
        output.append([char])
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):

    output = ""
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output += " "
        output += char
        output += " "
      else:
        output += char
    return output




  def _is_chinese_char(self, cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or
        (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x20000 and cp <= 0x2A6DF) or
        (cp >= 0x2A700 and cp <= 0x2B73F) or
        (cp >= 0x2B740 and cp <= 0x2B81F) or
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x2F800 and cp <= 0x2FA1F)):
      return  True

    return False

  def _clean_text(self, text):
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)





class FullTokenizer(object):
  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, vocab=self.vocab)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      if preserve_token(token, self.vocab):
        split_tokens.append(token)
        continue
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_ids_to_tokens(self, ids):
    return [self.inv_vocab[id] for id in ids]

  def convert_tokens_to_ids(self, tokens):
    return [self.vocab[token] for token in tokens]




class WordpieceTokenizer(object):

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    text = convert_to_unicode(text)

    output_tokens = []

    t = text.strip().split() if text else []
    for token in t:
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue
      sub_tokens = []
      is_bad = False
      start = 0

      len_of_chars = len(chars)
      while start < len_of_chars:
        cur_substr = None
        end = len_of_chars
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr:
          sub_tokens.append(cur_substr)
        else:
          is_bad = True
          break
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_punctuation(char):
  cp = ord(char)
  char_list = [(33,47),(58,64),(91,96),(123,126)]
  flags=[]
  for low,upper in char_list:
    if cp>=low and cp<=upper:
      flags.append(1)
    else:
      flags.append(0)
  if sum(flags)!=0:
    return True
  if re.match(r'^P',unicodedata.category(char)):
    return True
  return False


def _is_whitespace(char):
  white_space = [" ","\t","\n","\r"]
  if char in white_space:
    return True
  if unicodedata.category(char) == "Zs":
    return True
  return False


def _is_control(char):
  control = ["\t","\n","\r"]
  if char in control:
    return False
  if unicodedata.category(char) in ("Cc", "Cf"):
    return True
  return False


