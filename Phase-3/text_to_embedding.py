import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertModel


class TextEmbeddingUsingBert():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("loading BERT Model")
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        print("BERT Model loaded")

    def add_special_tokens(self, text):
        return "[CLS] " + text + " [SEP]"

    def tokenize(self, marked_text):
        return self.tokenizer.tokenize(marked_text)
    
    def convert_tokens_to_ids(self, tokenized_text):
        return self.tokenizer.convert_tokens_to_ids(tokenized_text)
    
    def text_to_embedding(self, text):
        marked_text = self.add_special_tokens(text)
        tokenized_text = self.tokenize(marked_text)
        indexed_tokens = self.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = torch.tensor([indexed_tokens])
        self.model.eval()

        with torch.no_grad():
            output = self.model(indexed_tokens)
            last_hidden_state = output[1]
            text_embedding = last_hidden_state
        
        return text_embedding

    

