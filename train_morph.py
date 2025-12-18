import torch.nn as nn
import tqdm, torch
from model_morph import RNN_morph
from collections import defaultdict
from use_conllulib import CoNLLUReader
from torch.utils.data import TensorDataset, DataLoader


def pad_tensor(X, max_len, char_mode=True):
  res = torch.full((len(X), max_len), 0)
  for (i, row) in enumerate(X) :
    x_len = min(max_len, len(X[i]))
    res[i,:x_len] = torch.LongTensor(X[i][:x_len])
  return res

def fit(model, epochs, train_loader, dev_loader):
  criterion = nn.CrossEntropyLoss(ignore_index=0)
  optimizer = torch.optim.Adam(model.parameters()) 
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (X, y) in tqdm.tqdm(train_loader):
      optimizer.zero_grad()
      y_hat = model(X)
      
      B, T, C = y_hat.shape
      y_hat = y_hat.reshape(B*T, C)
      y = y.reshape(B*T)

      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()  
    print(f"{epoch+1}/{epochs} ")
    print("train_loss = {:.4f}".format(total_loss / len(train_loader.dataset)))
    print("dev_loss = {:.4f} dev_acc = {:.4f}".format(*perf(model, dev_loader, criterion)))

def perf(model, dev_loader, criterion):
  model.eval()
  total_loss = correct = total_tokens = 0
  for (X, y) in dev_loader:
    with torch.no_grad():
      y_hat = model(X) 
            
      B, T, C = y_hat.shape
      y_hat = y_hat.reshape(B*T, C)
      y = y.reshape(B*T)

      total_loss += criterion(y_hat, y)
      y_pred = y_hat.argmax(dim=-1)  
      mask = (y != 0)                # only consider real tokens
      correct += ((y_pred == y) * mask).sum().item()
      total_tokens += mask.sum().item()
      
  total = len(dev_loader.dataset)
  return total_loss / total, correct / total_tokens

def build_loader(in_enc, ends, out_enc, max_c, max_w, batch_size, batch_mode=True):
    if batch_size:
        in_enc_tensor = pad_tensor(in_enc, max_c)
        ends_tensor = pad_tensor(ends, max_w)
        out_enc_tensor = pad_tensor(out_enc, max_w)
        dataset = TensorDataset(in_enc_tensor, ends_tensor, out_enc_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train_mode)
        return dataloader, in_enc, ends, out_enc
    else:
        return in_enc, ends_out_enc

if __name__ == "__main__" : 
  train_loader, wordvocab, tagvocab = read_corpus(filename="../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train", wordvocab=None, tagvocab=None, max_len=40, batch_size=32, train_mode=True, batch_mode=True)
  num_embeddings= len(wordvocab) 
  output_size= len(tagvocab) 
  hidden_size=200
  embedding_dim=250
  hp = {
    "model_type": "GRU", 
    "embedding_dim": embedding_dim, 
    "hidden_size": hidden_size}
  
  dev_loader, _, _ = read_corpus(filename="../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev", wordvocab=wordvocab, tagvocab=tagvocab, max_len=40, batch_size=32, train_mode=False, batch_mode=True)
  model = RNN_morph(hidden_size=hidden_size, output_size=output_size, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
  fit(model=model, epochs=15, train_loader=train_loader, dev_loader=dev_loader)
  
  torch.save({"wordvocab": dict(wordvocab), 
              "tagvocab": dict(tagvocab), 
              "model_params": model.state_dict(),
              "hyperparams": hp}, "model.pt")