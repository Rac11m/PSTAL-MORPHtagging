import typing as T
from conllu import parse_incr
from collections import defaultdict, Counter


def load_char(in_file: str) -> T.Tuple(T.List, T.List, T.List):
    '''
    function that takes as an input a conllu file path and return a tuple of
    three lists (chars, in_enc, ends)
    '''
    sent_list = []
    for sent in parse_incr(open(in_file, encoding='UTF-8')):
        sent_list.append(sent.metadata['text'])

    chars = []
    in_enc = []
    ends = []
    vocab = ["<pad>"] + sorted(set("".join(sent_list)))    
    for i, v in enumerate(vocab):
        if v == ' ':
            vocab[i] = '<esp>'
        
    for s in sent_list[:1]:
        c = ["<pad>"] + [w if w != " " else "<esp>" for w in s]
        chars.append(c)

        char_to_int = [vocab.index(w) for w in c]
        in_enc.append(char_to_int)
    
        end = []
        for idx, w in enumerate(s): 
            if w == ' ':
              end.append(idx-1)     
        ends.append(end)
    
    return chars, in_enc, ends 