import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics

from .data_utils import SentenceREDataset, get_idx2tag, get_type2idx 
from .model import SentenceRE

here = os.path.dirname(os.path.abspath(__file__))

def test(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file
    typeset_file = hparams.typeset_file
    model_file = hparams.model_file

    test_file = hparams.test_file
    max_len = hparams.max_len
    validation_batch_size = hparams.validation_batch_size

    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    type2idx = get_type2idx(typeset_file)
    hparams.typeset_size = len(type2idx)
    model = SentenceRE(hparams).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    test_dataset = SentenceREDataset(test_file, tagset_path=tagset_file, typeset_path=typeset_file,
                                     pretrained_model_path=pretrained_model_path,
                                     max_len=max_len)
    val_loader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False)
    model.eval()

    with torch.no_grad():
        tags_true = []
        tags_pred = []
        for val_i_batch, val_sample_batched in enumerate(tqdm(val_loader, desc='Test')):
            token_ids = val_sample_batched['token_ids'].to(device)
            token_type_ids = val_sample_batched['token_type_ids'].to(device)
            attention_mask = val_sample_batched['attention_mask'].to(device)
            e1_mask = val_sample_batched['e1_mask'].to(device)
            e2_mask = val_sample_batched['e2_mask'].to(device)
            type_mask = val_sample_batched['type_mask'].to(device)
            tag_ids = val_sample_batched['tag_id']
            logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask, type_mask)
            pred_tag_ids = logits.argmax(1)
            tags_true.extend(tag_ids.tolist())
            tags_pred.extend(pred_tag_ids.tolist())

        print(metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys()), target_names=list(idx2tag.values())))
        f1 = metrics.f1_score(tags_true, tags_pred, average='weighted')
        precision = metrics.precision_score(tags_true, tags_pred, average='weighted')
        recall = metrics.recall_score(tags_true, tags_pred, average='weighted')
        accuracy = metrics.accuracy_score(tags_true, tags_pred)