import numpy as np
import pickle


with open('./translator/AAtoT.pkl', 'rb') as f:
    AA_to_Token = pickle.load(f)
with open('./translator/TtoAA.pkl', 'rb') as f:
    Token_to_AA = pickle.load(f)
with open('./translator/TtoAA_wS.pkl', 'rb') as f:
    Token_to_AA_with_special_tokens = pickle.load(f)


num_tokens = len(Token_to_AA_with_special_tokens)


def tokenize_P(P):
    length_P = 100
    SOS_P = np.array([2])
    EOS = np.array([8])
    PAD = np.array([9])
    X = np.array(list(map(lambda x: AA_to_Token[x], P)))
    X = X[:100]
    X = np.concatenate((SOS_P, X, EOS, np.repeat(PAD, length_P - len(X))))
    return X


def untokenize(tokenized):
    return "".join(Token_to_AA[idx] for idx in tokenized)


def untokenize_with_special_tokens(tokenized):
    return "".join(Token_to_AA_with_special_tokens[idx] for idx in tokenized)