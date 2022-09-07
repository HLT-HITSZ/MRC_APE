import re
from transformers import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
from transformers import RobertaTokenizer
from transformers import  LongformerTokenizer
max_co_occur_words_num = 46




root_dir = './data/'
processed_root_dir = './data/processed/'


special_tokens_yi = ['[TAB]', '[LINE]',
                          '[EQU]', '[URL]', '[NUM]',
                          '[SPE]', '<sep>','[q]'] 
special_tokens_dict_yi = {'additional_special_tokens': special_tokens_yi}


tokenizer = RobertaTokenizer.from_pretrained('./longformer-base/')
tokenizer.model_max_length =4098
tokenizer.add_special_tokens(special_tokens_dict_yi)


def preprocess(sent):
    sent = sent.replace('<tab>', ' [TAB] ')
    sent = re.sub('[+-]{3,}', ' [LINE] ', sent).strip()
    sent = re.sub('={2,}', ' [EQU] ', sent).strip()
    sent = re.sub('_{2,}', ' [LINE] ', sent).strip()
    sent = re.sub('<[^>]+>[^>]+>', ' [URL] ', sent).strip()
    sent = re.sub('[0-9]+\.[0-9]+', ' [NUM] ', sent).strip()
    sent = re.sub('(Äî‚)+', ' [SPE] ', sent).strip()

    sent = re.sub(' +', ' ', sent).strip()


    token_list = tokenizer.tokenize(sent)


    token_list = token_list[:140]
    token_sent = ' '.join(token_list) 

    return token_sent

max_len = 0
sent_len_list = []

char_counter = Counter()

for data_file in ['dev.txt', 'test.txt', 'train.txt']:
    with open(root_dir + data_file, 'r') as fp:
        raw_sample_list = fp.read().split('\n\n')
        print(data_file + ':' + str(len(raw_sample_list)))

    with open(processed_root_dir + data_file, 'w') as fp:
        sample_dict_list = []
        for raw_sample in raw_sample_list:
            if raw_sample == '':
                continue
            line_list = raw_sample.split('\n')
            sample_dict = {'review': {'sent_ids': [], 'sents': [], 'bio_tag': [], 'pair_tag': [], 'sub_id': None},
                           'reply': {'sent_ids': [], 'sents': [], 'bio_tag': [], 'pair_tag': [], 'sub_id': None},
                           'graph': set()}

            rev_idx = -1
            rep_idx = -1
            total_idx = -1
            for idx, line in enumerate(line_list):
                line = re.sub('\.\t', ' .\t', line).strip()
                line = re.sub('!\t', ' !\t', line).strip()
                line = re.sub('\?\t', ' ?\t', line).strip()
                tmp = line.strip().split('\t')

                char_counter += Counter(tmp[0])

                text_type = tmp[3].lower()
                if text_type == 'review':
                    if rev_idx >= 99:
                        continue
                    else:
                        rev_idx += 1
                        total_idx += 1
                else:
                    if rep_idx >= 99:
                        continue
                    else:
                        rep_idx += 1
                        total_idx += 1

                sample_dict[text_type]['sent_ids'].append(total_idx)
                sample_dict[text_type]['sents'].append(tmp[0])
                sample_dict[text_type]['bio_tag'].append(tmp[1])
                sample_dict[text_type]['pair_tag'].append(tmp[2])
                sample_dict[text_type]['sub_id'] = tmp[4]
            sample_dict_list.append(sample_dict)



        for sample_dict in sample_dict_list:
            review_dict, reply_dict = sample_dict['review'], sample_dict['reply']
            sub_id = review_dict['sub_id']
            for token_sent, bio_tag, pair_tag in zip(review_dict['sents'], \
                                                     review_dict['bio_tag'], \
                                                     review_dict['pair_tag']):
                token_sent = preprocess(token_sent)
                fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    token_sent, bio_tag, pair_tag, 'Review', sub_id))

            fp.write('\n')

            for token_sent, bio_tag, pair_tag in zip(reply_dict['sents'], \
                                                     reply_dict['bio_tag'], \
                                                     reply_dict['pair_tag']):
                token_sent = preprocess(token_sent)
                fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    token_sent, bio_tag, pair_tag, 'Reply', sub_id))

            fp.write('\n\n')

pass
