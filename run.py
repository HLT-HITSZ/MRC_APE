import torch, os, time, random, sys, json
import numpy as np
import logging
import torch.optim as optim
import torch.nn as nn

sys.path.append('./utils')
# from trans_module import TransitionModel, BertEncoderE
from models import BERT_BiLSTM_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
# from evaluation import evaluat
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from bmes_decode import extract_arguments, get_arg_span, spans_to_tags,extract_span_arguments,extract_span_arguments_yi

from config import get_config

config = get_config()
import datetime
from bmes_decode import extract_flat_spans

now = datetime.datetime.now()
now_time_string = "{:0>4d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}_{:0>5d}".format(
    now.year, now.month, now.day, now.hour, now.minute, now.second, config.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_path = './saved_models'
save_path = os.path.join(save_path, now_time_string)
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    print("save_path exists!!")
    exit(1)
with open(os.path.join(save_path, "config.json"), "w") as fp:
    json.dump(config.__dict__, fp)

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
# fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setFormatter(formatter)
# logger.addHandler(ch) # output to terminal
logger.addHandler(fh)  # output to file

tags2id = {'O': 0, 'B-Review': 1, 'I-Review': 2, 'E-Review': 3, 'S-Review': 4,
           'B-Reply': 1, 'I-Reply': 2, 'E-Reply': 3, 'S-Reply': 4,
           'B': 1, 'I': 2, 'E': 3, 'S': 4}

def load_data_new_sample(file_path):

    sample_list_task1_for_review = []
    sample_list_task1_for_reply = []
    sample_list_task2_for_review_dir = []
    sample_list_task2_for_reply_dir = []

    with open(file_path, 'r') as fp:
        rr_pair_list = fp.read().split('\n\n\n')
        for rr_pair in rr_pair_list:
            if rr_pair == '':
                continue
            review, reply = rr_pair.split('\n\n')

            sample_review = {'sentences': [], 'bio_tags': [],
                             'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in review.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_review['sentences'].append(sent)
                sample_review['bio_tags'].append(bio_tag)
                sample_review['pair_tags'].append(pair_tag)
                sample_review['text_type'] = text_type
                sample_review['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_review['bio_tags']]

            review_spans=get_arg_span(tags_ids)

            sample_review['arg_spans'] = review_spans

            seq_len = len(tags_ids)

            review_start_positions = []
            review_end_positions = []
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in review_spans:
                review_start_positions.append(start)
                review_end_positions.append(end)
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            start_labels = torch.LongTensor([(1 if idx in review_start_positions else 0) for idx in range(
                seq_len)]) 
            end_labels = torch.LongTensor([(1 if idx in review_end_positions else 0) for idx in range(
                seq_len)])  
            sample_review['match_labels'] = match_labels
            sample_review['start_labels'] = start_labels
            sample_review['end_labels'] = end_labels

            sample_review['tag']="task1_review"

            sample_list_task1_for_review.append(sample_review)

            sample_reply = {'sentences': [], 'bio_tags': [],
                            'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in reply.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_reply['sentences'].append(sent)
                sample_reply['bio_tags'].append(bio_tag)
                sample_reply['pair_tags'].append(pair_tag)
                sample_reply['text_type'] = text_type
                sample_reply['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_reply['bio_tags']]


            reply_spans = get_arg_span(tags_ids)

            sample_reply['arg_spans'] = reply_spans

            seq_len = len(tags_ids)

            reply_start_positions = []
            reply_end_positions = []
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in reply_spans:
                reply_start_positions.append(start)
                reply_end_positions.append(end)
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            start_labels = torch.LongTensor([(1 if idx in reply_start_positions else 0) for idx in
                                             range(
                                                 seq_len)])
            end_labels = torch.LongTensor([(1 if idx in reply_end_positions else 0) for idx in
                                           range(
                                               seq_len)]) 

            sample_reply['match_labels'] = match_labels
            sample_reply['start_labels'] = start_labels
            sample_reply['end_labels'] = end_labels


            sample_reply['tag'] = "task1_reply"
            sample_list_task1_for_reply.append(sample_reply)

            rev_arg_2_rep_arg_dict = {}
            for rev_arg_span in sample_review['arg_spans']:
                rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                rev_arg_2_rep_arg_dict[rev_arg_span] = []
                for rep_arg_span in sample_reply['arg_spans']:
                    rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                    if rev_arg_pair_id == rep_arg_pair_id:
                        rev_arg_2_rep_arg_dict[rev_arg_span].append(rep_arg_span)
            sample_review['rev_arg_2_rep_arg_dict'] = rev_arg_2_rep_arg_dict


            rep_seq_len = len(sample_reply['bio_tags'])



            for rev_arg_span, rep_arg_spans in rev_arg_2_rep_arg_dict.items():

                pair_reply_start_positions = []
                pair_reply_end_positions = []
                pair_match_labels = torch.zeros([rep_seq_len, rep_seq_len], dtype=torch.long)
                for start, end in rep_arg_spans:
                    pair_reply_start_positions.append(start)
                    pair_reply_end_positions.append(end)
                    if start >= rep_seq_len or end >= rep_seq_len:
                        continue
                    pair_match_labels[start, end] = 1

                pair_start_labels = torch.LongTensor([(1 if idx in pair_reply_start_positions else 0) for idx in range(rep_seq_len)])
                pair_end_labels = torch.LongTensor([(1 if idx in pair_reply_end_positions else 0) for idx in range(rep_seq_len)])


                sample_review_dir_temp={}
                sample_review_dir_temp['review_sentences']=sample_review['sentences']
                sample_review_dir_temp['reply_sentences'] = sample_reply['sentences']
                sample_review_dir_temp['match_labels']=pair_match_labels
                sample_review_dir_temp['start_labels'] = pair_start_labels
                sample_review_dir_temp['end_labels'] = pair_end_labels

                sample_review_dir_temp['tag'] ="task2_review"

                temp_rr_dict={}
                tags = spans_to_tags(rep_arg_spans, rep_seq_len)
                temp_rr_dict[rev_arg_span] = tags
                sample_review_dir_temp['rr_arg_dict']=temp_rr_dict

                sample_list_task2_for_review_dir.append(sample_review_dir_temp)

            rep_arg_2_rev_arg_dict = {}


            for rep_arg_span in sample_reply['arg_spans']:
                rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                rep_arg_2_rev_arg_dict[rep_arg_span] = []
                for rev_arg_span in sample_review['arg_spans']:
                    rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                    if rep_arg_pair_id == rev_arg_pair_id:
                        rep_arg_2_rev_arg_dict[rep_arg_span].append(rev_arg_span)
            sample_reply['rep_arg_2_rev_arg_dict'] = rep_arg_2_rev_arg_dict




            rev_seq_len = len(sample_review['bio_tags'])


            for rep_arg_span, rev_arg_spans in rep_arg_2_rev_arg_dict.items():

                pair_review_start_positions = []
                pair_review_end_positions = []
                pair_match_labels = torch.zeros([rev_seq_len, rev_seq_len], dtype=torch.long)
                for start, end in rev_arg_spans:
                    pair_review_start_positions.append(start)
                    pair_review_end_positions.append(end)
                    if start >= rev_seq_len or end >= rev_seq_len:
                        continue
                    pair_match_labels[start, end] = 1

                pair_start_labels = torch.LongTensor([(1 if idx in pair_review_start_positions else 0) for idx in range( rev_seq_len)])
                pair_end_labels = torch.LongTensor([(1 if idx in pair_review_end_positions else 0) for idx in range(rev_seq_len)])


                sample_reply_dir_temp = {}
                sample_reply_dir_temp['review_sentences'] = sample_review['sentences']
                sample_reply_dir_temp['reply_sentences'] = sample_reply['sentences']
                sample_reply_dir_temp['match_labels'] = pair_match_labels
                sample_reply_dir_temp['start_labels'] = pair_start_labels
                sample_reply_dir_temp['end_labels'] = pair_end_labels

                sample_reply_dir_temp['tag'] = "task2_reply"

                temp_rr_dict = {}
                tags = spans_to_tags(rev_arg_spans, rev_seq_len)
                temp_rr_dict[rep_arg_span] = tags
                sample_reply_dir_temp['rr_arg_dict']= temp_rr_dict


                sample_list_task2_for_reply_dir.append(sample_reply_dir_temp)

    return sample_list_task1_for_review,sample_list_task1_for_reply,sample_list_task2_for_review_dir,sample_list_task2_for_reply_dir


def load_data(file_path):
    sample_list = []
    with open(file_path, 'r') as fp:
        rr_pair_list = fp.read().split('\n\n\n')
        for rr_pair in rr_pair_list:
            if rr_pair == '':
                continue
            review, reply = rr_pair.split('\n\n')

            sample_review = {'sentences': [], 'bio_tags': [],
                             'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in review.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_review['sentences'].append(sent)
                sample_review['bio_tags'].append(bio_tag)
                sample_review['pair_tags'].append(pair_tag)
                sample_review['text_type'] = text_type
                sample_review['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_review['bio_tags']]

            review_spans=get_arg_span(tags_ids)

            sample_review['arg_spans'] = review_spans

            seq_len = len(tags_ids)

            review_start_positions = []
            review_end_positions = []
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in review_spans:
                review_start_positions.append(start)
                review_end_positions.append(end)
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            start_labels = torch.LongTensor([(1 if idx in review_start_positions else 0) for idx in range(
                seq_len)])  
            end_labels = torch.LongTensor([(1 if idx in review_end_positions else 0) for idx in range(
                seq_len)])  
            sample_review['match_labels'] = match_labels
            sample_review['start_labels'] = start_labels
            sample_review['end_labels'] = end_labels

            sample_reply = {'sentences': [], 'bio_tags': [],
                            'pair_tags': [], 'text_type': None, 'sub_ids': [], 'arg_spans': []}
            for line in reply.strip().split('\n'):
                sent, bio_tag, pair_tag, text_type, sub_id = line.strip().split('\t')
                sample_reply['sentences'].append(sent)
                sample_reply['bio_tags'].append(bio_tag)
                sample_reply['pair_tags'].append(pair_tag)
                sample_reply['text_type'] = text_type
                sample_reply['sub_ids'] = sub_id
            tags_ids = [tags2id[t] for t in sample_reply['bio_tags']]


            reply_spans = get_arg_span(tags_ids)

            sample_reply['arg_spans'] = reply_spans

            seq_len = len(tags_ids)

            reply_start_positions = []
            reply_end_positions = []
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in reply_spans:
                reply_start_positions.append(start)
                reply_end_positions.append(end)
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            start_labels = torch.LongTensor([(1 if idx in reply_start_positions else 0) for idx in
                                             range(
                                                 seq_len)]) 
            end_labels = torch.LongTensor([(1 if idx in reply_end_positions else 0) for idx in
                                           range(
                                               seq_len)]) 

            sample_reply['match_labels'] = match_labels
            sample_reply['start_labels'] = start_labels
            sample_reply['end_labels'] = end_labels




            rev_arg_2_rep_arg_dict = {}
            for rev_arg_span in sample_review['arg_spans']:
                rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                rev_arg_2_rep_arg_dict[rev_arg_span] = []
                for rep_arg_span in sample_reply['arg_spans']:
                    rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                    if rev_arg_pair_id == rep_arg_pair_id:
                        rev_arg_2_rep_arg_dict[rev_arg_span].append(rep_arg_span)
            sample_review['rev_arg_2_rep_arg_dict'] = rev_arg_2_rep_arg_dict


            rep_seq_len = len(sample_reply['bio_tags'])

            rev_arg_2_rep_arg_dict_sem = {}
            for rev_arg_span, rep_arg_spans in rev_arg_2_rep_arg_dict.items():

                pair_reply_start_positions = []
                pair_reply_end_positions = []
                pair_match_labels = torch.zeros([rep_seq_len, rep_seq_len], dtype=torch.long)
                for start, end in rep_arg_spans:
                    pair_reply_start_positions.append(start)
                    pair_reply_end_positions.append(end)
                    if start >= rep_seq_len or end >= rep_seq_len:
                        continue
                    pair_match_labels[start, end] = 1

                pair_start_labels = torch.LongTensor([(1 if idx in pair_reply_start_positions else 0) for idx in range(rep_seq_len)])
                pair_end_labels = torch.LongTensor([(1 if idx in pair_reply_end_positions else 0) for idx in range(rep_seq_len)])

                temp_dict={}
                temp_dict['match_labels'] = pair_match_labels
                temp_dict['start_labels'] = pair_start_labels
                temp_dict['end_labels'] = pair_end_labels

                rev_arg_2_rep_arg_dict_sem[rev_arg_span] =temp_dict

            sample_review['rev_arg_2_rep_arg_dict_sem'] = rev_arg_2_rep_arg_dict_sem


            rev_arg_2_rep_arg_tags_dict = {}
            for rev_arg_span, rep_arg_spans in rev_arg_2_rep_arg_dict.items():
                tags = spans_to_tags(rep_arg_spans, rep_seq_len)
                rev_arg_2_rep_arg_tags_dict[rev_arg_span] = tags
            sample_review['rev_arg_2_rep_arg_tags_dict'] = rev_arg_2_rep_arg_tags_dict

            rep_arg_2_rev_arg_dict = {}
            for rep_arg_span in sample_reply['arg_spans']:
                rep_arg_pair_id = int(sample_reply['pair_tags'][rep_arg_span[0]].split('-')[-1])
                rep_arg_2_rev_arg_dict[rep_arg_span] = []
                for rev_arg_span in sample_review['arg_spans']:
                    rev_arg_pair_id = int(sample_review['pair_tags'][rev_arg_span[0]].split('-')[-1])
                    if rep_arg_pair_id == rev_arg_pair_id:
                        rep_arg_2_rev_arg_dict[rep_arg_span].append(rev_arg_span)
            sample_reply['rep_arg_2_rev_arg_dict'] = rep_arg_2_rev_arg_dict



            rev_seq_len = len(sample_review['bio_tags'])

            rep_arg_2_rev_arg_dict_sem={}
            for rep_arg_span, rev_arg_spans in rep_arg_2_rev_arg_dict.items():

                pair_review_start_positions = []
                pair_review_end_positions = []
                pair_match_labels = torch.zeros([rev_seq_len, rev_seq_len], dtype=torch.long)
                for start, end in rev_arg_spans:
                    pair_review_start_positions.append(start)
                    pair_review_end_positions.append(end)
                    if start >= rev_seq_len or end >= rev_seq_len:
                        continue
                    pair_match_labels[start, end] = 1

                pair_start_labels = torch.LongTensor([(1 if idx in pair_review_start_positions else 0) for idx in range( rev_seq_len)])
                pair_end_labels = torch.LongTensor([(1 if idx in pair_review_end_positions else 0) for idx in range(rev_seq_len)])

                temp_dict={}
                temp_dict['match_labels'] = pair_match_labels
                temp_dict['start_labels'] = pair_start_labels
                temp_dict['end_labels'] = pair_end_labels

                rep_arg_2_rev_arg_dict_sem[rep_arg_span] = temp_dict
            sample_reply['rep_arg_2_rev_arg_dict_sem'] = rep_arg_2_rev_arg_dict_sem


            rep_arg_2_rev_arg_tags_dict = {}
            for rep_arg_span, rev_arg_spans in rep_arg_2_rev_arg_dict.items():
                tags = spans_to_tags(rev_arg_spans, rev_seq_len)
                rep_arg_2_rev_arg_tags_dict[rep_arg_span] = tags
            sample_reply['rep_arg_2_rev_arg_tags_dict'] = rep_arg_2_rev_arg_tags_dict

            sample_list.append({'review': sample_review,
                                'reply': sample_reply})
    return sample_list


def args_metric(true_args_list, pred_args_list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for true_args, pred_args in zip(true_args_list, pred_args_list):
        true_args_set = set(true_args)
        pred_args_set = set(pred_args)
        assert len(true_args_set) == len(true_args)
        assert len(pred_args_set) == len(pred_args)
        tp += len(true_args_set & pred_args_set)
        fp += len(pred_args_set - true_args_set)
        fn += len(true_args_set - pred_args_set)
    if tp + fp == 0:
        pre = tp / (tp + fp + 1e-10)
    else:
        pre = tp / (tp + fp)
    if tp + fn == 0:
        rec = tp / (tp + fn + 1e-10)
    else:
        rec = tp / (tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre * rec) / (pre + rec + 1e-10)
    else:
        f1 = (2 * pre * rec) / (pre + rec)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {'pre': pre, 'rec': rec, 'f1': f1, 'acc': acc}


def evaluate(model, data_list):
    data_len = len(data_list)
    model.eval()

    all_true_rev_args_list = []
    all_pred_rev_args_list = []
    all_true_rep_args_list = []
    all_pred_rep_args_list = []

    all_true_arg_pairs_list = []

    all_pred_arg_pairs_list = []
    all_pred_arg_pairs_list_from_rev = []
    all_pred_arg_pairs_list_from_rep = []


    for batch_i in tqdm(range(data_len)):
        data_batch = data_list[batch_i :(batch_i + 1) ]


        review_para_tokens_list, review_tags_list = [], []
        reply_para_tokens_list, reply_tags_list = [], []

        review_match_labels, review_start_labels, review_end_labels = [], [], []
        reply_match_labels, reply_start_labels, reply_end_labels = [], [], []




        rev_arg_2_rep_arg_sems_list = []
        rep_arg_2_rev_arg_sems_list = []

        true_arg_pairs_list = []


        for sample in data_batch:

            review_para_tokens_list.append(sample['review']['sentences'])
            tags_ids = [tags2id[tag] for tag in sample['review']['bio_tags']]
            review_tags_list.append(tags_ids)
            # review for task1
            review_match_labels.append(sample['review']['match_labels'])
            review_start_labels.append(sample['review']['start_labels'])
            review_end_labels.append(sample['review']['end_labels'])
            # review for task2

            rep_arg_2_rev_arg_sems_list.append(sample['reply']['rep_arg_2_rev_arg_dict_sem'])

            reply_para_tokens_list.append(sample['reply']['sentences'])
            tags_ids = [tags2id[tag] for tag in sample['reply']['bio_tags']]
            reply_tags_list.append(tags_ids)
            #reply for task1
            reply_match_labels.append(sample['reply']['match_labels'])
            reply_start_labels.append(sample['reply']['start_labels'])
            reply_end_labels.append(sample['reply']['end_labels'])
            # reply for task2

            rev_arg_2_rep_arg_sems_list.append(sample['review']['rev_arg_2_rep_arg_dict_sem'])

            #task2 total

            arg_pairs = []
            for rev_arg, rep_args in sample['review']['rev_arg_2_rep_arg_dict'].items():
                for rep_arg in rep_args:
                    arg_pairs.append((rev_arg, rep_arg))
            true_arg_pairs_list.append(arg_pairs)



        with torch.no_grad():



            pred_rev_args_dict, pred_rep_args_dict,pred_pair_args_list, pred_pair_args_2_list = \
                model.predict_span(review_para_tokens_list, review_tags_list,
                              reply_para_tokens_list, reply_tags_list)

        true_rev_args_list_span = extract_span_arguments_yi(review_match_labels, review_start_labels, review_end_labels)
        all_true_rev_args_list.extend(true_rev_args_list_span)
        pred_rev_args_list_span = extract_span_arguments_yi(pred_rev_args_dict['review_span_preds'], pred_rev_args_dict['review_start_preds'], pred_rev_args_dict['review_end_preds'])
        all_pred_rev_args_list.extend(pred_rev_args_list_span)

        true_rep_args_list_span = extract_span_arguments_yi(reply_match_labels, reply_start_labels, reply_end_labels)
        all_true_rep_args_list.extend(true_rep_args_list_span)
        pred_rep_args_list_span = extract_span_arguments_yi(pred_rep_args_dict['reply_span_preds'],
                                                         pred_rep_args_dict['reply_start_preds'],
                                                         pred_rep_args_dict['reply_end_preds'])
        all_pred_rep_args_list.extend(pred_rep_args_list_span)

        all_true_arg_pairs_list.extend(true_arg_pairs_list) 

        pred_arg_pairs_list = []
        for pred_rep_args in pred_pair_args_list: 
            pred_arg_pairs = []

            for rev_arg, rep_args in pred_rep_args.items():
                for rep_arg, rep_arg_prob in zip(rep_args[0], rep_args[1]):
                    pred_arg_pairs.append((rev_arg, rep_arg))

            pred_arg_pairs_list.append(pred_arg_pairs)

        pred_arg_pairs_2_list = []
        for pred_rep_args_2 in pred_pair_args_2_list: 
            pred_arg_pairs = []

            for rep_arg, rev_args in pred_rep_args_2.items():
                for rev_arg, rev_arg_prob in zip(rev_args[0], rev_args[1]):
                    pred_arg_pairs.append((rev_arg, rep_arg))

            pred_arg_pairs_2_list.append(pred_arg_pairs)

        all_pred_arg_pairs_list.extend(
            [list(set(a + b)) for a, b in zip(pred_arg_pairs_list, pred_arg_pairs_2_list)]) 
        all_pred_arg_pairs_list_from_rev.extend([a for a in pred_arg_pairs_list])
        all_pred_arg_pairs_list_from_rep.extend([b for b in pred_arg_pairs_2_list])  

    args_pair_dict = args_metric(all_true_arg_pairs_list, all_pred_arg_pairs_list) 

    return  args_pair_dict



logger.warning('> training arguments:')
for arg in vars(config):
    logger.warning('>>> {0}: {1}'.format(arg, getattr(config, arg)))


train_list_task1_review,train_list_task1_reply, train_list_task2_review,train_list_task2_reply= \
    load_data_new_sample('./data/processed/train.txt.bioes')



dev_list = load_data('./data/processed/dev.txt.bioes')
test_list = load_data('./data/processed/test.txt.bioes')


train_list=train_list_task1_review+train_list_task1_reply+train_list_task2_review+train_list_task2_reply


train_len = len(train_list)
train_iter_len = (train_len // config.batch_size) + 1
if train_len % config.batch_size == 0:
    train_iter_len -= 1
num_training_steps = train_iter_len * config.epochs
num_warmup_steps = int(num_training_steps * config.warm_up)
logger.warning('Data loaded.')

logger.warning('Initializing model...')
model = BERT_BiLSTM_CRF(config)
model.cuda()
logger.warning('Model initialized.')


longformer_model_para = list(model.longformer.parameters())
lstm_para=list(model.am_bilstm.parameters())
other_model_para = list(set(model.parameters()) - set(longformer_model_para)-set(lstm_para))



longformer_base_encoder_lr=1e-5
lstm_para_lr=1e-3
finetune_lr=1e-3

optimizer_grouped_parameters = [
    {'params': [p for p in other_model_para if len(p.data.size()) > 1], 'weight_decay': config.weight_decay},
    {'params': [p for p in other_model_para if len(p.data.size()) == 1], 'weight_decay': 0.0},
    {'params': longformer_model_para, 'lr': longformer_base_encoder_lr},
    {'params': lstm_para, 'lr': lstm_para_lr}
]

optimizer = AdamW(optimizer_grouped_parameters, finetune_lr)

total_batch, early_stop = 0, 0
best_batch, best_f1 = 0, 0.0


random.shuffle(train_list)


for epoch_i in range(config.epochs):
    logger.warning("Running epoch: {}".format(epoch_i))
    loss_0, loss_1 = None, None
    last_loss_0, last_loss_1 = 0, 0
    bw_flag = False

    batch_id = 0
    for batch_i in tqdm(range(train_iter_len)):

        if True:

            model.train()
            train_batch = train_list[batch_i * config.batch_size:(batch_i + 1) * config.batch_size]
            # if len(train_batch) <= 1:
            #     continue

            para_tokens_list= []
            match_labels, start_labels,end_labels = [], [],[]

            para_tokens_list_for_2 = []

            rr_arg_pair_list=[]

            sample_tags_list=[]




            tt=[]

            for sample in train_batch:
                sample_tags_list.append(sample['tag'])

                if "task1" in sample['tag']:
                    para_tokens_list.append(sample['sentences'])

                    para_tokens_list_for_2.append([])
                    rr_arg_pair_list.append({})

                    match_labels.append(sample['match_labels'])
                    start_labels.append(sample['start_labels'])
                    end_labels.append(sample['end_labels'])



                elif "task2" in sample['tag']:
                    para_tokens_list.append(sample['review_sentences'])
                    para_tokens_list_for_2.append(sample['reply_sentences'])

                    rr_arg_pair_list.append(sample['rr_arg_dict'])

                    match_labels.append(sample['match_labels'])
                    start_labels.append(sample['start_labels'])
                    end_labels.append(sample['end_labels'])




            loss = model(para_tokens_list,para_tokens_list_for_2,rr_arg_pair_list,match_labels,start_labels,end_labels,tag_list_o=sample_tags_list)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()


        total_batch += 1
        batch_id += 1

    # evaluate
    t_start = time.time()

    dev_args_pair_dict=evaluate(model, dev_list)

    t_end = time.time()
    total_f1 = dev_args_pair_dict['f1']
    if total_f1 > best_f1:
        early_stop = 0
        best_f1 = total_f1
        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.mdl'))
        logger.warning('*' * 20 + 'best' + '*' * 20)
        best_batch = total_batch
        logger.warning('*' * 20 + 'the performance in valid set...' + '*' * 20)
        logger.warning('running time: {}'.format(t_end - t_start))
        logger.warning('total batch: {}'.format(total_batch))
        logger.warning('total pair f1:\t{:.4f}'.format(
            dev_args_pair_dict['f1']))

        test_args_pair_dict = evaluate(
            model, test_list)
        logger.warning('*' * 20 + 'the performance in test set...' + '*' * 20)
        logger.warning('total pair f1:\t{:.4f}'.format(
            test_args_pair_dict['f1']))
