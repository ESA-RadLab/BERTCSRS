import torch
from torch import nn
from transformers import BertModel


class BertClassifierParallel(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2, sigma=True):
        """ Create the BertClassifier.
        Params:
          - hidden: the size of the hidden layer
          - model_type: the name of the BERT model from HuggingFace
          - droupout: rate of the dropout to be applied
    """
        super(BertClassifierParallel, self).__init__()
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        self.parallel = nn.Conv1d(1, 1, 5)
        self.pool = nn.MaxPool1d(10)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden + 24, 25)
        # self.linear2 = nn.Linear(50, 25)
        self.linear3 = nn.Linear(25, 1)
        if sigma:
            self.activation = nn.Sigmoid()
        self.sigma = sigma

    def forward(self, input_id, mask, text):
        """ The forward pass of the BERT classifier.
       Params:
          - input_id: input_id from the tokenizer
          - mask: mask generated by the tokenizer
       Returns:
          - final_layer: the final output for the classification purpose
          - bert_outputs: all the outputs of the BERT model (together with attention)
    """
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        parallel_output = self.parallel(text)
        pooled_output = bert_outputs['pooler_output']
        pp_output = self.pool(parallel_output)
        combined_layer = torch.cat([pooled_output, pp_output.squeeze(1)], dim=1)
        hidden_layer1 = self.linear1(self.dropout(self.relu(combined_layer)))
        # hidden_layer2 = self.linear2(self.dropout(self.relu(hidden_layer1)))
        hidden_layer3 = self.linear3(self.dropout(self.relu(hidden_layer1)))
        if self.sigma:
            final_layer = self.activation(hidden_layer3)
            return final_layer, bert_outputs
        else:
            return hidden_layer3, bert_outputs


class BertClassifier10(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2, sigma=True):
        """ Create the BertClassifier.
        Params:
          - hidden: the size of the hidden layer
          - model_type: the name of the BERT model from HuggingFace
          - droupout: rate of the dropout to be applied
    """
        super(BertClassifier10, self).__init__()
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        # self.bert.train()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden, 10)
        # self.linear2 = nn.Linear(50, 25)
        self.linear3 = nn.Linear(10, 1)
        if sigma:
            self.activation = nn.Sigmoid()
        self.sigma = sigma

    def forward(self, input_id, mask):
        """ The forward pass of the BERT classifier.
       Params:
          - input_id: input_id from the tokenizer
          - mask: mask generated by the tokenizer
       Returns:
          - final_layer: the final output for the classification purpose
          - bert_outputs: all the outputs of the BERT model (together with attention)
    """
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = bert_outputs['pooler_output']
        hidden_layer1 = self.linear1(self.dropout(self.relu(pooled_output)))
        # hidden_layer2 = self.linear2(self.dropout(self.relu(hidden_layer1)))
        hidden_layer3 = self.linear3(self.dropout(self.relu(hidden_layer1)))
        if self.sigma:
            final_layer = self.activation(hidden_layer3)
            return final_layer, bert_outputs
        else:
            return hidden_layer3, bert_outputs


class BertClassifierConv(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2, sigma=True):

        super(BertClassifierConv, self).__init__()
        self.relu = nn.ReLU()

        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        self.dropBert = nn.Dropout(dropout)

        self.convolution3 = nn.Conv1d(1, 3, 10)
        conv_size3 = hidden - 10 + 1
        self.pool3 = nn.AvgPool1d(conv_size3//10, stride=conv_size3//10)

        self.convolution5 = nn.Conv1d(1, 3, 20)
        conv_size5 = hidden - 20 + 1
        self.pool5 = nn.AvgPool1d(conv_size5//10, stride=conv_size5//10)

        self.convolution7 = nn.Conv1d(1, 3, 40)
        conv_size7 = hidden - 40 + 1
        self.pool7 = nn.AvgPool1d(conv_size7//10, stride=conv_size7//10)

        self.convolution9 = nn.Conv1d(1, 3, 80)
        conv_size9 = hidden - 80 + 1
        self.pool9 = nn.AvgPool1d(conv_size9//10, stride=conv_size9//10)

        self.convolution11 = nn.Conv1d(1, 3, 160)
        conv_size11 = hidden - 160 + 1
        self.pool11 = nn.AvgPool1d(conv_size11//10, stride=conv_size11//10)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(150, 20)
        self.dropout1 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(20, 1)

        if sigma:
            self.activation = nn.Sigmoid()
        self.sigma = sigma

    def forward(self, input_id, mask):

        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = bert_outputs['pooler_output']
        pooled_output = self.dropBert(self.relu(pooled_output.unsqueeze(1)))

        conv_layer3 = self.convolution3(pooled_output)
        pool_layer3 = self.pool3(self.relu(conv_layer3))
        flat_layer3 = self.flatten(pool_layer3)

        conv_layer5 = self.convolution5(pooled_output)
        pool_layer5 = self.pool5(self.relu(conv_layer5))
        flat_layer5 = self.flatten(pool_layer5)

        conv_layer7 = self.convolution7(pooled_output)
        pool_layer7 = self.pool7(self.relu(conv_layer7))
        flat_layer7 = self.flatten(pool_layer7)

        conv_layer9 = self.convolution9(pooled_output)
        pool_layer9 = self.pool9(self.relu(conv_layer9))
        flat_layer9 = self.flatten(pool_layer9)

        conv_layer11 = self.convolution11(pooled_output)
        pool_layer11 = self.pool11(self.relu(conv_layer11))
        flat_layer11 = self.flatten(pool_layer11)

        concat_layer = torch.concat((flat_layer3, flat_layer5, flat_layer7, flat_layer9, flat_layer11), dim=1)

        hidden_layer1 = self.linear1(concat_layer)
        hidden_layer1 = self.dropout1(self.relu(hidden_layer1))

        hidden_layer3 = self.linear3(hidden_layer1)

        if self.sigma:
            final_layer = self.activation(hidden_layer3)
            return final_layer, bert_outputs
        else:
            return hidden_layer3, bert_outputs

    def set_sigma(self, sigma):
        self.sigma = sigma


class BertClassifier25(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2, sigma=True):
        """ Create the BertClassifier.
        Params:
          - hidden: the size of the hidden layer
          - model_type: the name of the BERT model from HuggingFace
          - droupout: rate of the dropout to be applied
    """
        super(BertClassifier25, self).__init__()
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        # self.bert.train()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden, 25)
        # self.linear2 = nn.Linear(50, 25)
        self.linear3 = nn.Linear(25, 1)
        if sigma:
            self.activation = nn.Sigmoid()
        self.sigma = sigma

    def forward(self, input_id, mask):
        """ The forward pass of the BERT classifier.
       Params:
          - input_id: input_id from the tokenizer
          - mask: mask generated by the tokenizer
       Returns:
          - final_layer: the final output for the classification purpose
          - bert_outputs: all the outputs of the BERT model (together with attention)
    """
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = bert_outputs['pooler_output']
        hidden_layer1 = self.linear1(self.dropout(self.relu(pooled_output)))
        # hidden_layer2 = self.linear2(self.dropout(self.relu(hidden_layer1)))
        hidden_layer3 = self.linear3(self.dropout(self.relu(hidden_layer1)))
        if self.sigma:
            final_layer = self.activation(hidden_layer3)
            return final_layer, bert_outputs
        else:
            return hidden_layer3, bert_outputs


class BertClassifier50(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2, sigma=True):
        """ Create the BertClassifier.
        Params:
          - hidden: the size of the hidden layer
          - model_type: the name of the BERT model from HuggingFace
          - droupout: rate of the dropout to be applied
    """
        super(BertClassifier50, self).__init__()
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        # self.bert.train()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden, 50)
        # self.linear2 = nn.Linear(50, 25)
        self.linear3 = nn.Linear(50, 1)
        if sigma:
            self.activation = nn.Sigmoid()
        self.sigma = sigma

    def forward(self, input_id, mask):
        """ The forward pass of the BERT classifier.
       Params:
          - input_id: input_id from the tokenizer
          - mask: mask generated by the tokenizer
       Returns:
          - final_layer: the final output for the classification purpose
          - bert_outputs: all the outputs of the BERT model (together with attention)
    """
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = bert_outputs['pooler_output']
        hidden_layer1 = self.linear1(self.dropout(self.relu(pooled_output)))
        # hidden_layer2 = self.linear2(self.dropout(self.relu(hidden_layer1)))
        hidden_layer3 = self.linear3(self.dropout(self.relu(hidden_layer1)))
        if self.sigma:
            final_layer = self.activation(hidden_layer3)
            return final_layer, bert_outputs
        else:
            return hidden_layer3, bert_outputs


class BertClassifier5025(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2, sigma=True):
        """ Create the BertClassifier.
        Params:
          - hidden: the size of the hidden layer
          - model_type: the name of the BERT model from HuggingFace
          - droupout: rate of the dropout to be applied
    """
        super(BertClassifier5025, self).__init__()
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        # self.bert.train()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden, 50)
        self.linear2 = nn.Linear(50, 25)
        self.linear3 = nn.Linear(25, 1)
        if sigma:
            self.activation = nn.Sigmoid()
        self.sigma = sigma

    def forward(self, input_id, mask):
        """ The forward pass of the BERT classifier.
       Params:
          - input_id: input_id from the tokenizer
          - mask: mask generated by the tokenizer
       Returns:
          - final_layer: the final output for the classification purpose
          - bert_outputs: all the outputs of the BERT model (together with attention)
    """
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = bert_outputs['pooler_output']
        hidden_layer1 = self.linear1(self.dropout(self.relu(pooled_output)))
        hidden_layer2 = self.linear2(self.dropout(self.relu(hidden_layer1)))
        hidden_layer3 = self.linear3(self.dropout(self.relu(hidden_layer2)))
        if self.sigma:
            final_layer = self.activation(hidden_layer3)
            return final_layer, bert_outputs
        else:
            return hidden_layer3, bert_outputs


class BertClassifier2525(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2, sigma=True):
        """ Create the BertClassifier.
        Params:
          - hidden: the size of the hidden layer
          - model_type: the name of the BERT model from HuggingFace
          - droupout: rate of the dropout to be applied
    """
        super(BertClassifier2525, self).__init__()
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        # self.bert.train()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden, 25)
        self.linear2 = nn.Linear(25, 25)
        self.linear3 = nn.Linear(25, 1)
        if sigma:
            self.activation = nn.Sigmoid()
        self.sigma = sigma

    def forward(self, input_id, mask):
        """ The forward pass of the BERT classifier.
       Params:
          - input_id: input_id from the tokenizer
          - mask: mask generated by the tokenizer
       Returns:
          - final_layer: the final output for the classification purpose
          - bert_outputs: all the outputs of the BERT model (together with attention)
    """
        bert_outputs = self.bert(input_ids=input_id, attention_mask=mask)
        pooled_output = bert_outputs['pooler_output']
        hidden_layer1 = self.linear1(self.dropout(self.relu(pooled_output)))
        hidden_layer2 = self.linear2(self.dropout(self.relu(hidden_layer1)))
        hidden_layer3 = self.linear3(self.dropout(self.relu(hidden_layer2)))
        if self.sigma:
            final_layer = self.activation(hidden_layer3)
            return final_layer, bert_outputs
        else:
            return hidden_layer3, bert_outputs
