from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch
from torch import nn


class bert_ATE(BertPreTrainedModel):
    def __init__(self, config):
        super(bert_ATE, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 3)  # Maps hidden size to 3 output classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get the last hidden state from the BERT outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # Pass through the linear layer
        logits = self.classifier(last_hidden_state)  # Shape: [batch_size, seq_len, 3]

        if labels is not None:
            # Compute the loss
            loss = self.loss_fn(logits.view(-1, 3), labels.view(-1))  # Flatten both tensors
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


class bert_ABSA(BertPreTrainedModel):
    def __init__(self, config):
        super(bert_ABSA, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.3)  # Adding dropout with prob=0.3
        self.classifier = nn.Linear(config.hidden_size, 3)  # 3 sentiment classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Get the pooled output from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output  # Shape: [batch_size, hidden_size]

        pooled_output = self.dropout(pooled_output)

        # Pass the pooled output through the linear layer
        logits = self.classifier(pooled_output)  # Shape: [batch_size, 3]

        if labels is not None:
            # Compute the loss
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
