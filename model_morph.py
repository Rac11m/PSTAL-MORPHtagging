import torch 
import torch.nn as nn


class RNN_morph(nn.Module):

    def __init__(self, hidden_size: int, output_size: int, num_embeddings: int, embedding_dim: int):
        '''
        num_embeddings = |V_c| (vocab of characters)
        embedding_dim = d_c (embedding size for the corresponding char)
        '''
        super().__init__()
        self.PAD_ID = 0
        self.embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx=self.PAD_ID) 
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bias=False, bidirectional=False)
        self.dropout = nn.Dropout(0.1)
        self.decision = nn.Linear(hidden_size, output_size)

    
    def forward(self, in_enc, ends):
        embedding = self.embed(in_enc)
        rnn_out, _ = self.gru(embedding)
        ends = ends.unsqueeze(-1) # unsqueeze the last dim (2)
        ends = ends.expand(-1, -1, rnn_out.size(-1)) # rnn_out.size(-1) =  rnn_out.d_h
        word_repr = rnn_out.gather(dim=1, index=ends) 
        word_repr = self.dropout(word_repr)            

        return self.decision(word_repr)
