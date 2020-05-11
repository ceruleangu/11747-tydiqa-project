from urllib.parse import urlparse
import os
import logging
import json
from pathlib import Path
import torch
from torch.hub import _get_torch_home
from zipfile import ZipFile, is_zipfile
from filelock import FileLock
import requests
import shutil
from hashlib import sha256
import fnmatch
import boto3
import tempfile
from contextlib import contextmanager
import tqdm
import copy
from functools import partial, wraps
from botocore.config import Config
import sys


torch_cache_home = _get_torch_home()
default_cache_path = os.path.join(torch_cache_home, "transformers")

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
TRANSFORMERS_CACHE = Path(os.getenv("PYTORCH_TRANSFORMERS_CACHE",
                                    os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path))
    )
S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-multilingual-uncased":
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased":
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
}


class PretrainedConfig(object):

    pretrained_config_archive_map = {}
    model_type = ""

    def __init__(self, **kwargs):

        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.id2label = kwargs.pop("id2label", {i: "LABEL_{}".format(i) for i in range(self.num_labels)})
        self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        self.label2id = kwargs.pop("label2id", dict(zip(self.id2label.values(), self.id2label.keys())))
        self.label2id = dict((key, int(value)) for key, value in self.label2id.items())

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @property
    def num_labels(self):
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels):
        self._num_labels = num_labels
        self.id2label = {i: "LABEL_{}".format(i) for i in range(self.num_labels)}
        self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))
        self.label2id = dict((key, int(value)) for key, value in self.label2id.items())

    def save_pretrained(self, save_directory):

        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)


    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path, pretrained_config_archive_map = None, **kwargs
    ) :

        dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if not pretrained_config_archive_map :
            pretrained_config_archive_map = cls.pretrained_config_archive_map

        if pretrained_model_name_or_path in pretrained_config_archive_map:
            config_file = pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or \
                urlparse(pretrained_model_name_or_path) in ["http", "https", "s3"] :
            config_file = pretrained_model_name_or_path
        else:
            config_file = "/".join((S3_BUCKET_PREFIX, pretrained_model_name_or_path, CONFIG_NAME))

        try:

            resolved_config_file = cached_path(
                config_file,
                dir=dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )

            if resolved_config_file is None:
                raise EnvironmentError
            config_dict = cls._dict_from_json_file(resolved_config_file)

        except EnvironmentError:
            raise EnvironmentError

        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(config_file, resolved_config_file))

        return config_dict, kwargs


    @classmethod
    def from_dict(cls, config_dict, **kwargs):

        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_dict(self):

        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            write_string = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
            writer.write(write_string)


class BertConfig(PretrainedConfig):

    pretrained_config_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

def cached_path(
            url_or_filename,
            dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            user_agent=None,
            extract_compressed_file=False,
            force_extract=False,
            local_files_only=False,
    ):

    if not dir:
        dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(dir, Path):
        dir = str(dir)

    if urlparse(url_or_filename) in ["http", "https", "s3"] :
        output_path = get_from_cache(
            url_or_filename,
            dir=dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            local_files_only=local_files_only,
            )
    elif os.path.exists(url_or_filename):
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if extract_compressed_file:
        if not is_zipfile(output_path):
            return output_path

        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

    return output_path

def s3_etag(url, proxies=None):
    def split_s3_path(url):

        parsed = urlparse(url)
        if not parsed.netloc or not parsed.path:
            raise ValueError("bad s3 path {}".format(url))
        bucket_name = parsed.netloc
        s3_path = parsed.path
        if s3_path.startswith("/"):
            s3_path = s3_path[1:]
        return bucket_name, s3_path

    s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag

def get_from_cache(
        url,
        dir=None,
        force_download=False,
        proxies=None,
        etag_timeout=10,
        resume_download=False,
        user_agent=None,
        local_files_only=False,
):

    if not dir :
        dir = TRANSFORMERS_CACHE
    if isinstance(dir, Path):
        dir = str(dir)

    os.makedirs(dir, exist_ok=True)
    etag = None
    if not local_files_only:
        if url.startswith("s3://"):
            etag = s3_etag(url, proxies=proxies)
        else:
            try:
                response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
                if response.status_code == 200:
                    etag = response.headers.get("ETag")
            except (EnvironmentError, requests.exceptions.Timeout):
                pass
    url_hash = sha256(url.encode("utf-8"))
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    path = os.path.join(dir, filename)


    if etag is None:
        if os.path.exists(path):
            return path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(dir), filename + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(dir, matching_files[-1])
            else:
                if local_files_only:
                    raise ValueError
                return None

    if os.path.exists(path) and not force_download:
        return path

    lock_path = path + ".lock"
    with FileLock(lock_path):

        if resume_download:
            incomplete_path = path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=dir, delete=False)
            resume_size = 0

        with temp_file_manager() as temp_file:

            if url.startswith("s3://"):
                s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
                bucket_name, s3_path = urlparse(url).netloc,urlparse(url).path
                if s3_path.startswith("/"):
                    s3_path = s3_path[1:]
                s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

            else:
                GetHttp(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)

        os.rename(temp_file.name, path)

        logger.info("creating metadata file for %s", path)
        meta = {"url": url, "etag": etag}
        meta_path = path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return path

def GetHttp(url, temp_file, proxies=None, resume_size=0, user_agent=None):
    ua = "transformers/{}; python/{}".format(__version__, sys.version.split()[0])

    ua += "; torch/{}".format(torch.__version__)

    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    headers = {"user-agent": ua}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:
        return
    content_length = response.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    prog = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    )
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            prog.update(len(chunk))
            temp_file.write(chunk)
    prog.close()
