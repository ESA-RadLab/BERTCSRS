from torch import nn
from transformers import BertModel


class BertClassifierOld(nn.Module):
    """The classifier model. BERT with a classification output head.
  """

    def __init__(self, hidden, model_type, dropout=0.2):
        """ Create the BertClassifier.
        Params:
          - hidden: the size of the hidden layer
          - model_type: the name of the BERT model from HuggingFace
          - droupout: rate of the dropout to be applied
    """
        super(BertClassifierOld, self).__init__()
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden, 2)
        self.activation = nn.Softmax()

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
        hidden_layer1 = self.linear1(self.dropout(pooled_output))
        final_layer = self.activation(hidden_layer1)
        return final_layer, bert_outputs