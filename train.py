import fire
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

import pytorch_lightning as pl
from model import BertNER
from transformers import BertTokenizer

#####################################################################
# Data preprocessing
# reference: https://github.com/huggingface/transformers/issues/64
#####################################################################

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
SEP = "[SEP]"
MASK = "[MASK]"
CLS = "[CLS]"
max_len = 200

LABELS = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O', "X"]
ids_to_labels = {k:v for k, v in enumerate(LABELS)}
labels_to_ids = {v:k for k, v in enumerate(LABELS)}
num_labels = len(LABELS) - 1 # model can't output "X"

def convert_tokens_to_ids(tokens, pad=True):
    """Helper function
    """
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.LongTensor([token_ids])
    assert ids.size(1) < max_len
    if pad:
        padded_ids = torch.zeros(max_len).long()
        padded_ids[:ids.size(1)] = ids
        mask = torch.zeros(max_len).long()
        mask[0:ids.size(1)] = 1
        return padded_ids, mask
    else:
        return ids

def subword_tokenize(tokens, labels):
    """
    Helper function
    Segment each token into subwords while keeping track of
    token boundaries.
    Parameters
    ----------
    tokens: A sequence of strings, representing input tokens.
    Returns
    -------
    A tuple consisting of:
        - A list of subwords, flanked by the special symbols required
            by Bert (CLS and SEP).
        - An array of indices into the list of subwords, indicating
          that the corresponding subword is the start of a new
            token. For example, [1, 3, 4, 7] means that the subwords
            1, 3, 4, 7 are token starts, while all other subwords
            (0, 2, 5, 6, 8...) are in or at the end of tokens.
            This list allows selecting Bert hidden states that
            represent tokens, which is necessary in sequence
            labeling.
    """
    def flatten(list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = [CLS] + list(flatten(subwords)) + [SEP]
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    # X label described in Bert Paper section 4.3
    bert_labels = [[label] + (sublen-1) * ["X"] for sublen, label in zip(subword_lengths, labels)]
    bert_labels = ["O"] + list(flatten(bert_labels)) + ["O"]

    assert len(subwords) == len(bert_labels)
    assert len(subwords) <= 512
    return subwords, token_start_idxs, bert_labels

def subword_tokenize_to_ids(tokens, labels):
    """Segment each token into subwords while keeping track of token boundaries and convert subwords into IDs.
    Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
    """
    assert len(tokens) == len(labels)
    subwords, token_start_idxs, bert_labels = subword_tokenize(tokens, labels)
    subword_ids, mask = convert_tokens_to_ids(subwords)
    token_starts = torch.zeros(max_len)
    token_starts[token_start_idxs] = 1
    bert_labels = [labels_to_ids[label] for label in bert_labels]
    # X label described in Bert Paper section 4.3 is used for pading
    padded_bert_labels = torch.ones(max_len).long() * labels_to_ids["X"]
    padded_bert_labels[:len(bert_labels)] = torch.LongTensor(bert_labels)

    mask.require_grad = False
    return {
        "input_ids": subword_ids,
        "attention_mask": mask,
        "bert_token_starts": token_starts,
        "labels": padded_bert_labels
    }

class CoNLL(Dataset):
    """simple class to read raw dataset
    """
    def __init__(self, path="./data"):
        entries = open(path, "r").read().strip().split("\n\n")

        self.sentences, self.labels = [], []  # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = [line.split()[-1] for line in entry.splitlines()]
            self.sentences.append(words)
            self.labels.append(tags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i], self.labels[i]

def prepare_dataset(path='./data/valid.txt'):
    dataset = CoNLL(path)
    featurized_sentences = []
    for tokens, labels in dataset:
        features = subword_tokenize_to_ids(tokens, labels)
        featurized_sentences.append(features)

    def collate(featurized_sentences_batch):
        keys = ("input_ids", "attention_mask", "bert_token_starts", "labels")
        output = {key: torch.stack([fs[key] for fs in featurized_sentences_batch], dim=0) for key in keys}
        return output

    dataset = collate(featurized_sentences)
    return TensorDataset(*[dataset[k] for k in ("input_ids", "attention_mask", "labels")])


###############
# Training
###############
def train(batch_size=32, lr=5e-5, epoch=4):
    N_GPUs = torch.cuda.device_count()
    val_dataset = prepare_dataset('./data/valid.txt')
    train_dataset = prepare_dataset('./data/train.txt')

    sampler = RandomSampler(train_dataset)
    train_dataloader= DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)
    val_dataloader= DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    model = BertNER(num_labels, lr, train_dataloader, val_dataloader, ids_to_labels, labels_to_ids)
    trainer = pl.Trainer(
        fast_dev_run=False if N_GPUs > 0 else True,
        gpus=N_GPUs if N_GPUs != 0 else 0,
        distributed_backend="dp" if N_GPUs > 1 else None,
        max_epochs=epoch,
        #overfit_pct=0.01
    )
    trainer.fit(model)

if __name__ == "__main__":
    fire.Fire(train)
