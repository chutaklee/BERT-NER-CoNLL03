import numpy as np
import torch
import pytorch_lightning as pl
from transformers import AdamW, BertModel

from conlleval import evaluate

class BertNER(pl.LightningModule):
    def __init__(self, num_labels, lr, train_dataloader, val_dataloader, ids_to_labels, labels_to_ids):
        super(BertNER, self).__init__()
        self.num_labels = num_labels
        self.lr = lr

        self.model = BertModel.from_pretrained("bert-base-cased")
        self.classifier = torch.nn.Linear(768, num_labels) # model cant not output "X" labels
        self.dropout = torch.nn.Dropout(p=0.1)

        self.traindl, self.valdl = train_dataloader, val_dataloader # we can't overwrite self.train, self.train_dataloader
        self.ids_to_labels = ids_to_labels
        self.labels_to_ids = labels_to_ids

    def f1(self, y_true, y_pred):
        flatten = lambda l: [item for sublist in l for item in sublist]
        y_true = flatten(y_true)
        y_pred = flatten(y_pred)
        y_true = [self.ids_to_labels[l] for l in y_true]
        y_pred = [self.ids_to_labels[l] for l in y_pred]
        assert len(y_pred) == len(y_true)

        ids = [i for i, label in enumerate(y_true) if label != "X"]
        y_true_cleaned = [y_true[i] for i in ids]
        y_pred_cleaned = [y_pred[i] for i in ids]

        precision, recall, f1 = evaluate(y_true_cleaned, y_pred_cleaned)#, verbose=False)
        print(f"micro average precision: {precision}, recall: {recall}, f1: {f1}")
        return precision, recall, f1

    def forward(self, input_ids, attention_mask, labels=None):
        sequence_output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            # reference:
            # https://github.com/huggingface/transformers/blob/d5d7d886128732091e92afff7fcb3e094c71a7ec/src/transformers/modeling_bert.py#L1380-L1394
            loss_fct = torch.nn.CrossEntropyLoss()
            # X label described in Bert Paper section 4.3
            X = self.labels_to_ids["X"]
            not_X_mask = labels != X # since label of PAD is "X", attention mask is not needed

            # Only keep active parts of the loss
            active_loss = not_X_mask.view(-1)
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), scores

    def training_step(self, batch, batch_idx):
        loss, score = self.forward(*batch)
        tqdm_dict = {"train_loss": loss}
        return {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}

    def validation_step(self, batch, batch_idx):
        _, mask, labels = batch
        loss, score = self.forward(*batch)
        labels_pred = torch.argmax(score, dim=-1)
        return {
            "val_loss": loss,
            "y_true": labels,
            "y_pred": labels_pred,
            "mask": mask,
        }

    def validation_end(self, outputs):
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)

        y_true, y_pred = [], []
        for out in outputs:
            batch_y_true = out["y_true"].cpu().numpy().tolist()
            batch_y_pred = out["y_pred"].cpu().numpy().tolist()
            batch_seq_lens = out["mask"].cpu().numpy().sum(-1).tolist()
            for i, length in enumerate(batch_seq_lens):
                batch_y_true[i] = batch_y_true[i][:length]
                batch_y_pred[i] = batch_y_pred[i][:length]
            y_true += batch_y_true
            y_pred += batch_y_pred

        precision, recall, f1 = self.f1(y_true, y_pred)
        tqdm_dict = {
            "val_loss": val_loss,
            "f1": f1
        }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @pl.data_loader
    def train_dataloader(self):
        return self.traindl

    @pl.data_loader
    def val_dataloader(self):
        return self.valdl
