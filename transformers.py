import logging
import math
import os
import math
import re
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

import BertConfig

from utils import PreTrainedModel, prune_linear_layer


logger = logging.getLogger(__name__)
BertLayerNorm = torch.nn.LayerNorm

if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu

def swish(x):
    return x * torch.sigmoid(x)

def _gelu_python(x):

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


Function = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": F.tanh,
    "gelu_new": gelu_new,
}


def get_activation(activation_string):
    if activation_string in Function:
        return Function[activation_string]
    else:
        raise KeyError("function {} not found in Function mapping {}".format(activation_string, list(Function.keys())))



def load_weights(model, config, tf_checkpoint_path):

    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))

    variables = tf.train.list_variables(tf_path)

    names_arrays = []
    for name, shape in variables:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        paths = tf.train.load_variable(tf_path, name)
        names_arrays.append((name,paths))

    optimizer = ["adam_v", "adam_m", "AdamWeightDecayOptimizer",
                 "AdamWeightDecayOptimizer_1", "global_step"]

    for name, path in names_arrays:
        name = name.split("/")

        if any(n in optimizer for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            first_name = scope_names[0]
            second_name =scope_names[1]
            if first_name in ["kernel" , "gamma"]:
                pointer = getattr(pointer, "weight")
            elif first_name in ["output_bias" ,"beta"]:
                pointer = getattr(pointer, "bias")
            elif first_name == "output_weights":
                pointer = getattr(pointer, "weight")
            elif first_name == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, first_name)
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(second_name)
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(path)
        try:
            assert pointer.shape == path.shape
        except AssertionError as e:
            e.args += (pointer.shape, path.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(path)
    return model


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))




class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = [self.LayerNorm,self.dropout]

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs=None):
        if not input_ids:
            input_shape = input_ids.size()
        else:
            input_shape = inputs.size()[:-1]

        seq_length = input_shape[1]

        if input_ids:
            device = input_ids.device
        else:
            device = inputs.device

        if not position_ids:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if not token_type_ids:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if not inputs:
            inputs = self.word_embeddings(input_ids)
        position = self.position(position_ids)
        token_type = self.token_type(token_type_ids)

        embeddings = inputs
        embeddings += position
        embeddings += token_type
        for layer in self.layers:
            embeddings = layer(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size {} is not a multiple of the number of attention \
                heads {}" .format (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_attention_tops = config.num_attention_heads
        self.attention_top_size = int(config.hidden_size / config.num_attention_heads)
        self.all_top_size = self.num_attention_tops * self.attention_top_size

        self.key = nn.Linear(config.hidden_size, self.all_top_size)
        self.value = nn.Linear(config.hidden_size, self.all_top_size)
        self.query = nn.Linear(config.hidden_size, self.all_top_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,hidden_states,
        attention_mask=None, head_mask=None,
        encoder_hidden_states = None, encoder_attention_mask=None):

        mixed_query = self.query(hidden_states)

        if not encoder_hidden_states:
            mixed_key = self.key(hidden_states)
            mixed_value = self.value(hidden_states)
        else:
            mixed_key = self.key(encoder_hidden_states)
            mixed_value = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask

        query_dense = self.score_transpose(mixed_query)
        key_dense = self.score_transpose(mixed_key)

        scores = torch.matmul(query_dense, key_dense.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_top_size)

        if attention_mask:
            scores += attention_mask

        probs = self.dropout(nn.Softmax(dim=-1)(scores))

        if head_mask:
            probs = head_mask * probs

        value_dense = self.score_transpose(mixed_value)
        context_dense = torch.matmul(probs, value_dense)

        context_dense = context_dense.permute(0, 2, 1, 3).contiguous()
        new_shape = context_dense.size()[:-2] + (self.all_top_size,)
        context_dense = context_dense.view(*new_shape)

        if self.output_attentions:
            out = (context_dense, probs)
        else:
            out =(context_dense,)
        return out

    def socre_transpose(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_tops, self.attention_top_size)
        x = x.view(*new_x_shape)
        out = x.permute(0, 2, 1, 3)
        return out


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [self.dense,self.dropout]

    def forward(self, hidden_states, input_tensor):
        out = hidden_states
        for layer in self.layers:
            out = layer(out)
        out = self.LayerNorm(out + input_tensor)
        return out


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.selfAttention = SelfAttention(config)
        self.output = SelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.selfAttention.forward(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self_outputs[0]
        attention_output = self.output.forward(attention_output, hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def prune_heads(self, heads):
        if not heads:
            return
        mask = torch.ones(self.selfAttention.num_attention_tops, self.selfAttention.attention_top_size)
        unique_heads = set(heads)
        unique_heads -=  self.pruned_heads
        for head in unique_heads:
            sum_pruned =0
            for h in self.pruned_heads:
                if h<head:
                    sum_pruned += 1

            head = head - sum_pruned
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.selfAttention.query = prune_linear_layer(self.selfAttention.query, index)
        self.selfAttention.key = prune_linear_layer(self.selfAttention.key, index)
        self.selfAttention.value = prune_linear_layer(self.selfAttention.value, index)

        self.self.num_attention_tops = self.selfAttention.num_attention_tops - len(heads)
        self.self.all_top_size = self.selfAttention.attention_top_size * self.selfAttention.num_attention_tops
        self.pruned_heads = self.pruned_heads.union(heads)


class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if type(config.hidden_act)== str:
            self.intermediate_act_fn = Function[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.layers = [self.dense,self.intermediate_act_fn]

    def forward(self, hidden_states):
        out = hidden_states
        for layer in self.layers:
            out = layer(out)
        return out


class Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [self.dense, self.dropout]

    def forward(self, hidden_states, input_tensor):
        out = hidden_states
        for layer in self.layers:
            out = layer(out)
        out = self.LayerNorm(out + input_tensor)
        return out


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.cross_attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention.forward(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states:
            cross_attention_outputs = self.cross_attention.forward(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )

            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)

        return (layer_output,) + outputs


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,hidden_states,
        attention_mask=None,head_mask=None,
        encoder_hidden_states=None,encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = Function[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers=[self.dense,self.transform_act_fn,self.LayerNorm]

    def forward(self, hidden_states):
        out = hidden_states
        for layer in self.layers:
            out = layer(out)
        return out


class LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = PredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.decoder.bias = self.bias
        self.layers = [self.transform,self.decoder]

    def forward(self, hidden_states):
        out = hidden_states
        for layer in self.layers:
            out = layer(out)

        return out


class OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LMPredictionHead(config)

    def forward(self, sequence_output):
        scores = self.predictions.forward(sequence_output)
        return scores


class OnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.layers  = [self.predictions,self.seq_relationship]

    def forward(self, sequence_output, pooled_output):

        prediction_scores = self.layers[0](sequence_output)
        seq_relationship_score = self.layers[1](pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedModel(PreTrainedModel):

    config_class = BertConfig
    pretrained_model_archive_map = BertConfig.BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_weights
    base_model_prefix = "bert"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias:
            module.bias.data.zero_()

class BertModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.pool = BertPool(config)
        self.init_weights()

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):

        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_mask=None,
    ):

        if not input_ids  and not inputs_embeds :
            raise ValueError("Need input_ids or inputs_embeds")
        elif input_ids  and  inputs_embeds:
            raise ValueError("You cannot get both input_ids and inputs_embeds at one time")
        elif input_ids:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if not input_ids:

            device = input_ids.device
        else :
            device = inputs_embeds.device

        if not token_type_ids :
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if not attention_mask:
            attention_mask = torch.ones(input_shape, device=device)

        if attention_mask.dim() == 2:

            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )
                extended_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) \
                or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)
        extended_mask = (1.0 - extended_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if not encoder_mask :
                encoder_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_mask.dim() == 3:
                encoder_extended_mask = encoder_mask[:, None, :, :]
            elif encoder_mask.dim() == 2:
                encoder_extended_mask = encoder_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) \
                    or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_mask.shape
                    )
                )

            encoder_extended_mask = \
                encoder_extended_mask.to(
                dtype=next(self.parameters()).dtype
            )
            encoder_extended_mask = (1.0 - encoder_extended_mask) * -10000.0
        else:
            encoder_extended_mask = None

        if head_mask:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder.forward(
            embedding_output,
            attention_mask=extended_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pool.forward(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs

