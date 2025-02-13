from transformers import RobertaTokenizer, RobertaModel
import torch

# Load the CodeBERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

def get_code_embedding(code: str):
    inputs = tokenizer(code, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    # We take the mean of the token embeddings as the sentence representation
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.squeeze().detach().numpy()
