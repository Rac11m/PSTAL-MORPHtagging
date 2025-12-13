import typing as T
from conllu import parse_incr
from collections import defaultdict, Counter


def load_chars(in_file: str, train_mode: bool):
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
    vocab = ["<pad>", "<unk>"] + sorted(set("".join(sent_list)))    
    for i, v in enumerate(vocab):
        if v == ' ':
            vocab[i] = '<esp>'
        
    for s in sent_list[:1]:
        c = ["<pad>"] + [w if w != " " else "<esp>" for w in s]
        chars.append(c)

        if train_mode:
            char_to_int = [vocab.index(w) for w in c]
        else:
            char_to_int =  [vocab.index(w) if w in vocab else vocab.index("<unk>") for w in c]
        in_enc.append(char_to_int)
        
        end = []
        for idx, w in enumerate(s): 
            if w == ' ':
              end.append(idx-1)     
        ends.append(end)
    
    return chars, in_enc, ends 

def build_feats_dict(in_file: str):
    feats_values = {}
    for df in sorted(diff_feats):
        feats_values[df] = {"<N/A>": 0}
    sents = parse_incr(open(file_path, encoding="UTF-8"))
    for sent in sents:
        for tok in sent:
            if tok["feats"] is not None:
                for l in list(tok["feats"].keys()):
                    if tok["feats"][l] not in feats_values[l]:
                        feats_values[l][tok["feats"][l]] = 0
    for k,v in feats_values.items():
        for i,w in enumerate(feats_values[k]): 
            feats_values[k][w] = i
    
    return feats_values

def load_feats(in_file: str, feat: str, feat_dict: dict):
    feat_list = []
    for sent in parse_incr(open(in_file, encoding='UTF-8')):
        feat_sent = []
        for tok in sent:    
            if tok["feats"] is None:
                feat_sent.append("<N/A>")
            elif feat in tok["feats"].keys():
                feat_sent.append(tok["feats"][feat])
            else:
                feat_sent.append("<N/A>")
        feat_list.append(feat_sent)

    out_enc = []
    for fs in feat_list:
        out_enc.append([feats_values[feat][w] for w in fs])
    
    return out_enc

