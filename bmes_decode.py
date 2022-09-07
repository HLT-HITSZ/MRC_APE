from typing import Tuple, List
import numpy as np
import torch
tags2id = {'O': 0, 'B-Review': 1, 'I-Review': 2, 'E-Review': 3, 'S-Review': 4,
           'B-Reply': 1, 'I-Reply': 2, 'E-Reply': 3, 'S-Reply': 4,
           'B': 1, 'I': 2, 'E': 3, 'S': 4}
def spans_to_tags(spans, seq_len):
    tags = [tags2id['O']] * seq_len
    for span in spans:
        tags[span[0]] = tags2id['B']
        tags[span[0]:span[1]+1] = [tags2id['I']] * (span[1]-span[0]+1)
        if span[0] == span[1]:
            tags[span[0]] = tags2id['S']
        else:
            tags[span[0]] = tags2id['B']
            tags[span[1]] = tags2id['E']
    return tags


def get_arg_span(bioes_tags):
    start, end = None, None
    arguments = []
    in_entity_flag = False
    for idx, tag in enumerate(bioes_tags):
        if in_entity_flag == False:
            if tag == 1: # B
                in_entity_flag = True
                start = idx
            elif tag == 4: # S
                start = idx
                end = idx
                arguments.append((start, end))
                start = None
                end = None
        else:
            if tag == 0: # O
                in_entity_flag = False
                start = None
                end = None
            elif tag == 1: # B
                in_entity_flag = True
                start = idx
            elif tag == 3: # E
                in_entity_flag = False
                end = idx
                arguments.append((start, end))
                start = None
                end = None
            elif tag == 4: # S
                in_entity_flag = False
                start = idx
                end = idx
                arguments.append((start, end))
                start = None
                end = None
    return arguments




def extract_arguments(bioes_list):
    arguments_list = []
    for pred_tags in bioes_list:
        arguments = get_arg_span(pred_tags)
        arguments_list.append(arguments)
    return arguments_list

def extract_span_arguments_yi(match_labels,start_labels,end_labels):
    arguments_list = []
    for match_l, start_l, end_l in zip(match_labels,start_labels,end_labels):
        arguments = extract_flat_spans_yi( start_l, end_l,match_l)
        arguments_list.append(arguments)
    return arguments_list
def extract_span_arguments(match_labels,start_labels,end_labels):
    arguments_list = []
    for match_l, start_l, end_l in zip(match_labels,start_labels,end_labels):
        arguments = extract_flat_spans( start_l, end_l,match_l)
        arguments_list.append(arguments)
    return arguments_list

def extract_span_arguments_nested(match_labels,start_labels,end_labels):
    arguments_list = []
    for match_l, start_l, end_l in zip(match_labels,start_labels,end_labels):
        arguments = extract_flat_spans_nested( start_l, end_l,match_l)
        arguments_list.append(arguments)
    return arguments_list

class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]:
    """
    decode inputs to tags
    Args:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "S-LOC")]
        >>> bmes_decode(x)
        [{'term': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 2}]
    """
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        # correct labels
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            idx += 1
            continue
            # print("?")
            # raise Exception("Invalid Inputs")
    return tags

def extract_flat_spans_nested(start_pred, end_pred, match_pred,  pseudo_tag = "TAG"):
    seq_len = start_pred.size()[0]

    start_l_mask = [[1 for i in range(seq_len)]]
    end_l_mask = [[1 for i in range(seq_len)]]

    start_label_mask = torch.LongTensor(start_l_mask).cuda()
    end_label_mask = torch.LongTensor(end_l_mask).cuda()

    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    bsz, seq_len = start_label_mask.size()


    start_preds = start_pred.bool().unsqueeze(0).cuda()
    end_preds = end_pred.bool().unsqueeze(0).cuda()
    match_pred_s=match_pred.bool().unsqueeze(0).cuda()


    match_preds = (match_pred_s & start_preds.unsqueeze(-1).expand(-1, -1, seq_len) & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len) & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    match_pos_pairs = np.transpose(np.nonzero(match_preds.cpu().numpy())).tolist()
    return [(pos[1], pos[2]) for pos in match_pos_pairs]

def extract_flat_spans(start_pred, end_pred, match_pred,  pseudo_tag = "TAG"):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    pseudo_input = "a"

    label_mask=[1]*len(start_pred) #TODO

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_tag}"

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(entity.begin, entity.end-1) for entity in tags]


def extract_flat_spans_yi(start_pred, end_pred, match_pred):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    pseudo_input = "a"

    label_mask=[1]*len(start_pred) #TODO

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"I"
            else:
                bmes_labels[tmp_end] = f"S"

    tags = get_arg_span([tags2id[label] for label in bmes_labels])

    return tags