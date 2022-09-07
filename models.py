import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('./utils')

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence

import torch
from transformers import LongformerModel,LongformerConfig,LongformerTokenizer
from classifier import MultiNonLinearClassifier
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from bmes_decode import extract_arguments, get_arg_span, spans_to_tags,extract_span_arguments,extract_span_arguments_yi



class BERT_BiLSTM_CRF(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = config.layers
        self.hidden_size = config.hidden_size
        self.mlp_size = config.mlp_size
        self.dropout = nn.Dropout(p=config.dropout)
        self.scale_factor = config.scale_factor

        self.special_tokens_yi = ['[TAB]', '[LINE]',
                                  '[EQU]', '[URL]', '[NUM]',
                                  '[SPE]', '<sep>', '[q]']
        self.special_tokens_dict_yi = {'additional_special_tokens': self.special_tokens_yi}

        self.longtokenizer = LongformerTokenizer.from_pretrained(
            './longformer-base/')
        self.longtokenizer.add_special_tokens(self.special_tokens_dict_yi) 

        self.longformerconfig = LongformerConfig.from_pretrained(
            './longformer-base/')
        self.longformerconfig.attention_mode = 'sliding_chunks'
        self.longformerconfig.attention_window = [8,8,8,8,8,8,8,8,8,8,8,8]
        self.attentionwindow = self.longformerconfig.attention_window[0]

        self.longformer = LongformerModel.from_pretrained(
            './longformer-base/', config=self.longformerconfig)

        self.longformer.resize_token_embeddings(len(self.longtokenizer))


        self.am_bilstm = nn.LSTM(config.bert_output_size, config.hidden_size, \
                                 num_layers=1, bidirectional=config.is_bi, batch_first=True)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size*2, 1, config.mrc_dropout,
                                                       intermediate_hidden_size=128)#256

        self.span_loss_candidates = ["all", "pred_and_gold", "pred_gold_random", "gold"][0] 

        self.bce_loss = BCEWithLogitsLoss(reduction="none")

        self.weight_start = config.weight_start
        self.weight_end = config.weight_end
        self.weight_span = config.weight_span


    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels):

        batch_size, seq_len = start_logits.size()

        start_l_mask = torch.ones([batch_size, seq_len],dtype=torch.long)
        end_l_mask =  torch.ones([batch_size, seq_len],dtype=torch.long)

        start_label_mask = torch.LongTensor(start_l_mask).cuda()
        end_label_mask = torch.LongTensor(end_l_mask).cuda()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            elif self.span_loss_candidates == "pred_gold_random":
                gold_and_pred = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
                data_generator = torch.Generator()
                data_generator.manual_seed(0)
                random_matrix = torch.empty(batch_size, seq_len, seq_len).uniform_(0, 1)
                random_matrix = torch.bernoulli(random_matrix, generator=data_generator).long()
                random_matrix = random_matrix.cuda()
                match_candidates = torch.logical_or(
                    gold_and_pred, random_matrix
                )
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)

        return start_loss, end_loss, match_loss

    def bert_emb_for_task1(self, para_tokens_list):
        para_len_list = [len(para) for para in para_tokens_list]
        max_para_len = max(para_len_list)

        question_tokens = '[q]'
        question_tokens_cls_sep = self.longtokenizer.cls_token + ' ' + question_tokens + ' ' + self.longtokenizer.sep_token 
        question_ids = self.longtokenizer.convert_tokens_to_ids(question_tokens_cls_sep.split(' '))
        question_length = len(question_ids)

        sent_tokens_list = [sent for para in para_tokens_list for sent in para]
        sent_length_list = [len(sent.split(' ')) for para in para_tokens_list for sent in para]
        passage_tokens = ' '.join(sent_tokens_list)
        passage_tokens_cls_sep = self.longtokenizer.cls_token + ' ' + passage_tokens + ' ' + self.longtokenizer.sep_token

        sent_ids = self.longtokenizer.convert_tokens_to_ids(passage_tokens_cls_sep.split(' '))  # !

        question_sents_ids = [question_ids + sent_ids]

        qs_ids_padding_list, qs_mask_list, max_len = self.padding_and_mask_with_return(question_sents_ids) 
        qs_ids_padding_tensor = torch.tensor(qs_ids_padding_list).cuda()
        qs_mask_tensor = torch.tensor(qs_mask_list).cuda()

        _, global_att_mask = self.padding_and_mask_to_max_lenth([question_ids], max_len) 
        global_att_mask_tensor = torch.tensor(global_att_mask).cuda()

        try:

            longformer_outputs = self.longformer(qs_ids_padding_tensor,
                                                 attention_mask=qs_mask_tensor,
                                                 global_attention_mask=global_att_mask_tensor)
        except:
            import traceback
            traceback.print_exc()

        last_hidden_state = longformer_outputs[0] 
        last_hidden_state_for_s = last_hidden_state[:, (question_length + 1):-1, :]

        sen_emb_list = []
        start_index = 0
        end_index = 0
        for i in range(0, len(sent_length_list)):
            end_index = start_index + sent_length_list[i]
            last_h_temp = last_hidden_state_for_s[:, start_index:end_index, :]
            sen_emb_list.append(last_h_temp.mean(dim=-2))
            start_index = end_index

        sent_emb = torch.cat(sen_emb_list, dim=0)

        sent_emb = self.dropout(sent_emb)
        return sent_emb

    def bert_emb_for_task2(self, para_tokens_list,argument_para_tokens_list):
        para_len_list = [len(para) for para in para_tokens_list]
        max_para_len = max(para_len_list)

        argument_tokens=' '.join(argument_para_tokens_list)
        argument_tokens_cls_sep=self.longtokenizer.cls_token +' '+argument_tokens+' '+self.longtokenizer.sep_token
        argument_ids=self.longtokenizer.convert_tokens_to_ids(argument_tokens_cls_sep.split(' '))
        argument_length=len(argument_ids)

        sent_tokens_list = [sent for para in para_tokens_list for sent in para]
        sent_length_list = [len(sent.split(' ')) for para in para_tokens_list for sent in para]
        passage_tokens=' '.join(sent_tokens_list)
        passage_tokens_cls_sep=self.longtokenizer.sep_token +' '+passage_tokens+' '+self.longtokenizer.sep_token 
        sent_ids = self.longtokenizer.convert_tokens_to_ids(passage_tokens_cls_sep.split(' '))

        pair_ids=[argument_ids+sent_ids]


        pair_ids_padding_list, pair_mask_list,max_len = self.padding_and_mask_with_return(pair_ids)
        pair_ids_padding_tensor = torch.tensor(pair_ids_padding_list).cuda()
        pair_mask_tensor = torch.tensor(pair_mask_list).cuda()

        _,global_att_mask=self.padding_and_mask_to_max_lenth([argument_ids],max_len)
        global_att_mask_tensor=torch.tensor(global_att_mask).cuda()

        try:

            longformer_outputs = self.longformer(pair_ids_padding_tensor,
                                                 attention_mask=pair_mask_tensor,global_attention_mask=global_att_mask_tensor) 
        except:
            import traceback
            traceback.print_exc()

        last_hidden_state = longformer_outputs[0] 

        last_hidden_state_for_s = last_hidden_state[:, (argument_length+1):-1, :]

        sen_emb_list = []
        start_index = 0
        end_index = 0
        for i in range(0, len(sent_length_list)):
            end_index = start_index + sent_length_list[i]
            last_h_temp = last_hidden_state_for_s[:, start_index:end_index, :]
            sen_emb_list.append(last_h_temp.mean(dim=-2))
            start_index = end_index

        sent_emb = torch.cat(sen_emb_list, dim=0)

        sent_emb = self.dropout(sent_emb) 
        return sent_emb




    def am_tagging_span_for_task1(self, para_tokens_list, mode):





        sent_num_list = [len(para) for para in para_tokens_list]
        sent_emb = self.bert_emb_for_task1(para_tokens_list) 

        para_emb = torch.split(sent_emb, sent_num_list, 0)

        para_emb_packed = pack_sequence(para_emb, enforce_sorted=False)
        para_lstm_out_packed, (h, c) = self.am_bilstm(para_emb_packed)
        para_lstm_out_padded, _ = pad_packed_sequence(para_lstm_out_packed, batch_first=True)  


        para_lstm_out = para_lstm_out_padded
        batch_size, seq_len, hid_size = para_lstm_out.size()

        start_logits = self.start_outputs(para_lstm_out).squeeze(
            -1)  
        end_logits = self.end_outputs(para_lstm_out).squeeze(-1)  

        start_extend = para_lstm_out.unsqueeze(2).expand(-1, -1, seq_len, -1)

        end_extend = para_lstm_out.unsqueeze(1).expand(-1, seq_len, -1, -1)

        span_matrix = torch.cat([start_extend, end_extend], 3)


        span_logits = self.span_embedding(span_matrix).squeeze(-1) 



        return start_logits,end_logits,span_logits

    def am_tagging_span_for_task2(self, rev_para_tokens_list,rep_para_tokens_list, arg_pair_sems_list,mode='train'):


        sent_num_list = [len(para) for para in rep_para_tokens_list]

        arg_num_list = []


        temp_arg_list = []
        for batch_i, pred_arguments_labeldict in enumerate(arg_pair_sems_list):  
            for rev_arg_span, label_dict in pred_arguments_labeldict.items():



                temp_argu_o = rev_para_tokens_list[batch_i][rev_arg_span[0]:rev_arg_span[1] + 1] 
                temp_arg_list.append(temp_argu_o)
                arg_num_list.append(sent_num_list[0])

        para_lstm_out_list = []

        for arg in temp_arg_list:
            sent_emb = self.bert_emb_for_task2(rep_para_tokens_list, arg)
            para_emb = torch.split(sent_emb, sent_num_list, 0)
            para_emb_packed = pack_sequence(para_emb, enforce_sorted=False)

            para_lstm_out_packed, (h, c) = self.am_bilstm(para_emb_packed)
            para_lstm_out_padded, _ = pad_packed_sequence(para_lstm_out_packed, batch_first=True)
            para_lstm_out_list.append(para_lstm_out_padded)

        try:
            lstm_out_cat = torch.cat(para_lstm_out_list, dim=0)
        except:
            import traceback
            traceback.print_exc()

        para_lstm_out = lstm_out_cat 
        batch_size, seq_len, hid_size = para_lstm_out.size()


        start_logits = self.start_outputs(para_lstm_out).squeeze(
            -1) 
        end_logits = self.end_outputs(para_lstm_out).squeeze(-1) 

        start_extend = para_lstm_out.unsqueeze(2).expand(-1, -1, seq_len, -1)

        end_extend = para_lstm_out.unsqueeze(1).expand(-1, seq_len, -1, -1)

        span_matrix = torch.cat([start_extend, end_extend], 3)

        span_logits = self.span_embedding(span_matrix).squeeze(-1) 
        return start_logits, end_logits, span_logits


    def am_tagging_with_task1_for_task2(self, rev_para_tokens_list, rep_para_tokens_list, arg_list_from_task1,
                              mode='train'):

        sent_num_list = [len(para) for para in rep_para_tokens_list]

        arg_num_list = []

        temp_arg_list = []
        for batch_i, pred_arguments in enumerate(arg_list_from_task1):
            for rev_arg_span in pred_arguments:
                temp_argu_o = rev_para_tokens_list[0][rev_arg_span[0]:rev_arg_span[1] + 1] 
                temp_arg_list.append(temp_argu_o) 

                arg_num_list.append(sent_num_list[0]) 

        para_lstm_out_list = []

        for arg in temp_arg_list:
            sent_emb = self.bert_emb_for_task2(rep_para_tokens_list, arg)

            para_emb = torch.split(sent_emb, sent_num_list, 0)
            para_emb_packed = pack_sequence(para_emb, enforce_sorted=False)

            para_lstm_out_packed, (h, c) = self.am_bilstm(para_emb_packed)
            para_lstm_out_padded, _ = pad_packed_sequence(para_lstm_out_packed, batch_first=True)
            para_lstm_out_list.append(para_lstm_out_padded)

        try:
            lstm_out_cat = torch.cat(para_lstm_out_list, dim=0)
        except:
            import traceback
            traceback.print_exc()


        para_lstm_out = lstm_out_cat 
        batch_size, seq_len, hid_size = para_lstm_out.size()

        start_logits = self.start_outputs(para_lstm_out).squeeze(
            -1) 
        end_logits = self.end_outputs(para_lstm_out).squeeze(-1)

        start_extend = para_lstm_out.unsqueeze(2).expand(-1, -1, seq_len, -1)

        end_extend = para_lstm_out.unsqueeze(1).expand(-1, seq_len, -1, -1)

        span_matrix = torch.cat([start_extend, end_extend], 3)

        span_logits = self.span_embedding(span_matrix).squeeze(-1) 

        return start_logits, end_logits, span_logits 

    def forward(self, para_tokens_list_o,para_tokens_list_for_2_o,rr_arg_pair_list_o,
                     match_labels_o, start_labels_o, end_labels_o,tag_list_o):

        total_loss_all=0

        for para_tokens,para_tokens_2,rr_arg_pair,tag ,start_labels,end_labels,match_labels in zip(para_tokens_list_o,para_tokens_list_for_2_o,rr_arg_pair_list_o,tag_list_o,start_labels_o,end_labels_o,match_labels_o):

            para_tokens_list=[para_tokens]
            para_tokens_list_for_2=[para_tokens_2]
            rr_arg_pair_list=[rr_arg_pair]



            if tag=="task1_review" or tag=="task1_reply":

                start_logits, end_logits, span_logits= self.am_tagging_span_for_task1(para_tokens_list,mode="train")

            elif tag=="task2_review" :
                # review
                start_logits, end_logits, span_logits\
                    = self.am_tagging_span_for_task2(para_tokens_list,para_tokens_list_for_2,rr_arg_pair_list,mode="train")

            elif tag == "task2_reply":
                start_logits, end_logits, span_logits \
                    = self.am_tagging_span_for_task2(para_tokens_list_for_2, para_tokens_list, rr_arg_pair_list,mode="train")


            start_loss, end_loss, match_loss = self.compute_loss(start_logits,
                                                                 end_logits, span_logits,
                                                                 start_labels.expand(1,
                                                                                        -1).cuda(),
                                                                 end_labels.expand(1,
                                                                                      -1).cuda(),
                                                                 match_labels.expand(1, -1,
                                                                                        -1).cuda())
            total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

            total_loss_all=total_loss_all+total_loss



        total_loss_all_pingjun=total_loss_all/len(para_tokens_list_o)

        return total_loss_all_pingjun

    def predict_span(self, review_para_tokens_list, review_tags_list,

                     reply_para_tokens_list, reply_tags_list):


        # review
        review_start_logits, review_end_logits, review_span_logits = self.am_tagging_span_for_task1(
            review_para_tokens_list,
             mode="test")
        review_start_preds, review_end_preds, review_span_preds = F.sigmoid(review_start_logits) > 0.5, F.sigmoid(
            review_end_logits) > 0.5, F.sigmoid(review_span_logits) > 0.5

        pred_rev_args_dict = {}
        pred_rev_args_dict['review_start_preds'] = review_start_preds
        pred_rev_args_dict['review_end_preds'] = review_end_preds
        pred_rev_args_dict['review_span_preds'] = review_span_preds

        # reply
        reply_start_logits, reply_end_logits, reply_span_logits = self.am_tagging_span_for_task1(reply_para_tokens_list,

                                                                                                 mode="test")

        reply_start_preds, reply_end_preds, reply_span_preds = F.sigmoid(reply_start_logits) > 0.5, F.sigmoid(
            reply_end_logits) > 0.5, F.sigmoid(reply_span_logits) > 0.5

        pred_rep_args_dict = {}
        pred_rep_args_dict['reply_start_preds'] = reply_start_preds
        pred_rep_args_dict['reply_end_preds'] = reply_end_preds
        pred_rep_args_dict['reply_span_preds'] = reply_span_preds

        pred_rev_args_list = extract_span_arguments_yi(pred_rev_args_dict['review_span_preds'],
                                                       pred_rev_args_dict['review_start_preds'],
                                                       pred_rev_args_dict['review_end_preds'])
        pred_rep_args_list = extract_span_arguments_yi(pred_rep_args_dict['reply_span_preds'],
                                                       pred_rep_args_dict['reply_start_preds'],
                                                       pred_rep_args_dict['reply_end_preds'])

        test_rev_args_list = []

        for iitem in pred_rev_args_list:
            for arg in iitem:
                test_rev_args_list.append(arg)
        if test_rev_args_list == []:
            pred_args_pair_dict_list = [{} for t in pred_rev_args_list]
        else:
            try:
                # review
                review_start_logits, review_end_logits, review_span_logits \
                    = self.am_tagging_with_task1_for_task2(review_para_tokens_list, reply_para_tokens_list,
                                                           pred_rev_args_list, mode="test")

                review_start_preds, review_end_preds, review_span_preds = F.sigmoid(
                    review_start_logits) > 0.5, F.sigmoid(
                    review_end_logits) > 0.5, F.sigmoid(review_span_logits) > 0.5

            except:
                import traceback
                traceback.print_exc()

            review_span_preds_list = [i for i in review_span_preds]
            review_start_preds_list = [i for i in review_start_preds]
            review_end_preds_list = [i for i in review_end_preds]

            pred_pair_rep_args_list = extract_span_arguments_yi(review_span_preds_list, review_start_preds_list,
                                                                review_end_preds_list)

            # true_rev_args_list = extract_arguments(review_tags_list)
            pred_args_pair_dict_list = []
            i = 0
            for true_arguments in pred_rev_args_list:
                pred_args_pair_dict = {}
                for args in true_arguments:
                    pred_args_pair_dict[args] = (pred_pair_rep_args_list[i], \
                                                 [1] * len(pred_pair_rep_args_list[i]))
                    i += 1
                pred_args_pair_dict_list.append(pred_args_pair_dict)

        # reply
        test_rep_args_list = []
        # true_rep_args_list = extract_arguments(reply_tags_list)
        for iitem in pred_rep_args_list:
            for arg in iitem:
                test_rep_args_list.append(arg)
        if test_rep_args_list == []:
            pred_args_pair_dict_2_list = [{} for t in pred_rep_args_list]
        else:
            try:
                # reply
                reply_start_logits, reply_end_logits, reply_span_logits = self.am_tagging_with_task1_for_task2(
                    reply_para_tokens_list, review_para_tokens_list, pred_rep_args_list,
                    mode="test")

                # reply_start_preds, reply_end_preds, reply_span_preds = reply_start_logits > 0, reply_end_logits > 0, reply_span_logits > 0
                reply_start_preds, reply_end_preds, reply_span_preds = F.sigmoid(reply_start_logits) > 0.5, F.sigmoid(
                    reply_end_logits) > 0.5, F.sigmoid(reply_span_logits) > 0.5


            except:
                import traceback
                traceback.print_exc()

            rebuttal_span_preds_list = [i for i in reply_span_preds]
            rebuttal_start_preds_list = [i for i in reply_start_preds]
            rebuttal_end_preds_list = [i for i in reply_end_preds]

            pred_pair_rev_args_list = extract_span_arguments_yi(rebuttal_span_preds_list, rebuttal_start_preds_list,
                                                                rebuttal_end_preds_list)

            pred_args_pair_dict_2_list = []
            i = 0
            for true_arguments in pred_rep_args_list:
                pred_args_pair_dict = {}
                for args in true_arguments:
                    pred_args_pair_dict[args] = (pred_pair_rev_args_list[i], \
                                                 [1] * len(pred_pair_rev_args_list[i]))
                    i += 1
                pred_args_pair_dict_2_list.append(pred_args_pair_dict)

        return pred_rev_args_dict, pred_rep_args_dict, pred_args_pair_dict_list, pred_args_pair_dict_2_list

    def predict_span_for_task1(self, review_para_tokens_list, review_tags_list,

                     reply_para_tokens_list, reply_tags_list):

        # evaluate for task1:

        # review
        review_start_logits, review_end_logits, review_span_logits = self.am_tagging_span_for_task1(
            review_para_tokens_list,
             mode="test")

        review_start_preds, review_end_preds, review_span_preds = F.sigmoid(review_start_logits) > 0.5, F.sigmoid(
            review_end_logits) > 0.5, F.sigmoid(review_span_logits) > 0.5

        pred_rev_args_dict = {}
        pred_rev_args_dict['review_start_preds'] = review_start_preds
        pred_rev_args_dict['review_end_preds'] = review_end_preds
        pred_rev_args_dict['review_span_preds'] = review_span_preds

        # reply
        reply_start_logits, reply_end_logits, reply_span_logits = self.am_tagging_span_for_task1(reply_para_tokens_list,

                                                                                                 mode="test")
        reply_start_preds, reply_end_preds, reply_span_preds = F.sigmoid(reply_start_logits) > 0.5, F.sigmoid(
            reply_end_logits) > 0.5, F.sigmoid(reply_span_logits) > 0.5

        pred_rep_args_dict = {}
        pred_rep_args_dict['reply_start_preds'] = reply_start_preds
        pred_rep_args_dict['reply_end_preds'] = reply_end_preds
        pred_rep_args_dict['reply_span_preds'] = reply_span_preds


        return pred_rev_args_dict, pred_rep_args_dict




    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list
    def padding_and_mask_with_return(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list,max_len

    def padding_and_mask_to_max_lenth(self, ids_list,max_len):
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:

            mask = [1.] * len(ids) +[0.] * (max_len - len(ids)) 
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list


    def padding_matrix(self, matrix_tensor_list):
        seq_list=[]
        for matrix in matrix_tensor_list:
            seq=matrix.size()[0]
            seq_list.append(seq)

        max_seq_len = max(seq_list)

        new_matrix_list=[]
        for matrix in matrix_tensor_list:
            seq=matrix.size()[0]

            if seq<max_seq_len:

                o_t_list=torch.split(matrix,1,dim=0)
                p_o_t_list=[t[0] for t in o_t_list]
                left_num=max_seq_len-seq

                for i in range(left_num):
                    p_o_t_list.append(torch.zeros(max_seq_len,dtype=torch.long))

                new_matrix=pad_sequence(p_o_t_list,batch_first=True)
                new_matrix_list.append(new_matrix)

            else:
                new_matrix_list.append(matrix)

        padded_matrix=torch.stack(new_matrix_list)
        return padded_matrix