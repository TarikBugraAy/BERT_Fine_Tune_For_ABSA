from transformers import BertModel
import torch

class bert_ATE(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(bert_ATE, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)  # Maps hidden size to 3 output classes
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, tags_tensors, masks_tensors):
        # Get the last hidden state from the BERT outputs
        bert_outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors)
        last_hidden_state = bert_outputs[0]  # Shape: [batch_size, seq_len, hidden_size]

        # Pass through the linear layer
        linear_outputs = self.linear(last_hidden_state)  # Shape: [batch_size, seq_len, 3]

        if tags_tensors is not None:
            # Flatten tensors for CrossEntropyLoss
            tags_tensors = tags_tensors.view(-1)  # Shape: [batch_size * seq_len]
            linear_outputs = linear_outputs.view(-1, 3)  # Shape: [batch_size * seq_len, 3]

            # Compute the loss
            loss = self.loss_fn(linear_outputs, tags_tensors)
            return loss
        else:
            return linear_outputs


# class bert_ABSA(torch.nn.Module):
#     def __init__(self, pretrain_model):
#         super(bert_ABSA, self).__init__()
#         self.bert = BertModel.from_pretrained(pretrain_model)
#         self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)  # 3 sentiment classes
#         self.loss_fn = torch.nn.CrossEntropyLoss()

#     def forward(self, ids_tensors, lable_tensors, masks_tensors, segments_tensors):
#         # Get the outputs from BERT
#         outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
#         pooled_outputs = outputs[1]  # Extract pooled output for [CLS] token (shape: [batch_size, hidden_size])

#         # Pass the pooled output through the linear layer
#         linear_outputs = self.linear(pooled_outputs)  # Shape: [batch_size, 3]

#         if lable_tensors is not None:
#             # Compute the loss
#             loss = self.loss_fn(linear_outputs, lable_tensors)
#             return loss
#         else:
#             return linear_outputs

class bert_ABSA(torch.nn.Module):
    def __init__(self, pretrain_model, device):
        super(bert_ABSA, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.dropout = torch.nn.Dropout(p=0.3)  # Add dropout with prob=0.3 (tune as needed)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)  # 3 sentiment classes
        # weights = torch.tensor([1.5, 2.0, 0.5]).to(device)  # Move weights to the device
        # self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, lable_tensors, masks_tensors, segments_tensors):
        # Get the outputs from BERT
        outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
        pooled_outputs = outputs[1]  # Extract pooled output for [CLS] token (shape: [batch_size, hidden_size])

        pooled_outputs = self.dropout(pooled_outputs) # Drop the pooled outputs

        # Pass the pooled output through the linear layer
        linear_outputs = self.linear(pooled_outputs)  # Shape: [batch_size, 3]

        if lable_tensors is not None:
            # Compute the loss
            loss = self.loss_fn(linear_outputs, lable_tensors)
            return loss
        else:
            return linear_outputs
