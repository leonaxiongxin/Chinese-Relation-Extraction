import os
import re
import torch
import json

from .data_utils import MyTokenizer, get_idx2tag, convert_pos_to_mask
from .model import SentenceRE

here = os.path.dirname(os.path.abspath(__file__))


def predict(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file

    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = SentenceRE(hparams).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    predict_input_file = hparams.predict_input_file
    predict_output_file = hparams.predict_output_file
    entity_re_dict_file = hparams.entity_re_dict_file

    ENTITY_RE_DICT = json.load(open(entity_re_dict_file, encoding='utf-8'))

    tokenizer = MyTokenizer(pretrained_model_path)

    if predict_input_file and predict_output_file:
        item_list = json.load(open(predict_input_file))
        output_list = list()
        print('Prediction started')
        for item in item_list:
            entity_head = item["h"]
            entity_tail = item["t"]
            if entity_head["name"] == entity_tail["name"]:
                continue
            item_new = item.copy()
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
            encoded = tokenizer.bert_tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            token_type_ids = encoded['token_type_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            e1_mask = torch.tensor([convert_pos_to_mask(pos_e1, max_len=attention_mask.shape[1])]).to(device)
            e2_mask = torch.tensor([convert_pos_to_mask(pos_e2, max_len=attention_mask.shape[1])]).to(device)

            with torch.no_grad():
                logits = model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)[0]
                logits = logits.to(torch.device('cpu'))
            
            ENTITY_PAIR = '{}-{}'.format(entity_head["etype"], entity_tail["etype"])
            top_ids = logits.argsort(0, descending=True).tolist()
            # top_relation = idx2tag[logits.argmax(0).item()]
            # entity pair restriction
            for i, tag_id in enumerate(top_ids, start=1):
                if idx2tag[tag_id] == 'NA':
                    top_relation = 'NA'
                elif idx2tag[tag_id] in ENTITY_RE_DICT[ENTITY_PAIR]:
                    top_relation = idx2tag[tag_id]
                item_new['relation'] = top_relation
                output_list.append(item_new)
                break
        
        with open(predict_output_file, 'w') as fw:
            json.dump(output_list, fw, ensure_ascii=False, indent=4)
        print('Prediction finished')
    else:
        text = input("输入中文句子：")
        entity1 = input("句子中的实体1：")
        entity2 = input("句子中的实体2：")

        match_obj1 = re.search(entity1, text)
        match_obj2 = re.search(entity2, text)
        if match_obj1 and match_obj2:  # 姑且使用第一个匹配的实体的位置
            e1_pos = match_obj1.span()
            e2_pos = match_obj2.span()
            item = {
                'h': {
                    'name': entity1,
                    'pos': e1_pos
                },
                't': {
                    'name': entity2,
                    'pos': e2_pos
                },
                'text': text
            }
            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
            encoded = tokenizer.bert_tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            token_type_ids = encoded['token_type_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            e1_mask = torch.tensor([convert_pos_to_mask(pos_e1, max_len=attention_mask.shape[1])]).to(device)
            e2_mask = torch.tensor([convert_pos_to_mask(pos_e2, max_len=attention_mask.shape[1])]).to(device)

            with torch.no_grad():
                logits = model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)[0]
                logits = logits.to(torch.device('cpu'))
            print("最大可能的关系是：{}".format(idx2tag[logits.argmax(0).item()]))
            top_ids = logits.argsort(0, descending=True).tolist()
            for i, tag_id in enumerate(top_ids, start=1):
                print("No.{}：关系（{}）的可能性：{}".format(i, idx2tag[tag_id], logits[tag_id]))
        else:
            if match_obj1 is None:
                print('实体1不在句子中')
            if match_obj2 is None:
                print('实体2不在句子中')
