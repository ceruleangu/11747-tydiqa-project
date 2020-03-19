# coding=utf-8
# Copyright 2020 The Google Research Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Primary NN compute graph implemented in TensorFlow.

This module is a fairly thin wrapper over the BERT reference implementation.
"""

import collections

from absl import logging
from bert import modeling as bert_modeling
from bert import optimization as bert_optimization
import data

from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F


def TYDIQA(BertPreTrainedModel):
    "Create a QA moddl for tydi taks"

    def __init__(self, bert_config):
        super(BertPreTrainedModel, self).__init__(bert_config)
        self.num_answer_types = 5

        self.bert = BertModel(bert_config)

        self.qa_outputs = nn.Linear(bert_config.hidden_size, 2) #we need to label start and end position
        self.answer_type_output_dense = nn.Linear(self.bert.pooler_output.size[-1].value, self.num_answer_types)

        self.init_weights()

    def forward(self,
                input_ids = None,
                attention_mask = None,
                token_type_ids = None,
                position_ids = None,
                head_mask = None,
                inputs_embeds = None,
                start_positions = None,
                end_positions = None,
                answer_types = None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim = -1) #split logits into two, with each of size [batch * seq_len]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Get the logits for the answer type prediction.
        answer_type_output_layer = self.bert.pooler_output
        answer_type_logits = self.answer_type_output_dense(answer_type_output_layer)

        #get sequence length
        seq_length = sequence_output.size(1)

        def compute_loss(logits, positions):
            one_hot_positions = F.one_hot(
                positions, num_classes = seq_length
            )
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(one_hot_positions * log_probs, dim = -1))
            return loss

        # Computes the loss for labels.
        def compute_label_loss(logits, labels):
            one_hot_positions = F.one_hot(
                labels, num_classes=self.num_answer_types
            )
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(one_hot_positions * log_probs, dim=-1))
            return loss

        start_loss = compute_loss(start_logits, start_positions)
        end_loss = compute_loss(end_logits, end_positions)

        answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

        total_loss = (start_loss + end_loss + answer_type_loss) / 3.0

        return start_logits, end_logits, answer_type_logits, total_loss




