# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 8:32
import copy
import json
import typing
import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



class CorpusPreprocess:
   
    @classmethod
    def process(cls,tokenizer,lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            prompt = jd['prompt']
            # response = jd['response']
            chosen = jd['chosen']
            rejected = jd['rejected']
            if chosen == rejected:
                continue
            D.append((prompt,chosen, rejected))
        return D



class TokenIdsMaker:
    @classmethod
    def trunction_ids(cls,a_ids: typing.List,b_ids: typing.List,max_seq_length,sptoken):
        while len(a_ids) + len(b_ids) > max_seq_length - len(sptoken) - 1:
            if len(a_ids) > len(b_ids):
                a_ids.pop(0)
            else:
                b_ids.pop(-1)

    @classmethod
    def process(cls, pair_data, tokenizer: PreTrainedTokenizer, sptoken, max_seq_length: int, src_max_length,
                dst_max_length):
        prompt, chosen, rejected = pair_data

        a_ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        b_ids1 = tokenizer.encode(chosen, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        b_ids2 = tokenizer.encode(rejected, truncation=True, max_length=max_seq_length, add_special_tokens=False)

        if src_max_length is not None and src_max_length > 0:
            a_ids = a_ids[:src_max_length]

        if dst_max_length is not None and dst_max_length > 0:
            b_ids1 = b_ids1[:dst_max_length]
            b_ids2 = b_ids2[:dst_max_length]

        a_ids1 = copy.deepcopy(a_ids)
        cls.trunction_ids(a_ids1, b_ids1, max_seq_length, sptoken)

        a_ids2 = copy.deepcopy(a_ids)
        cls.trunction_ids(a_ids2, b_ids2, max_seq_length, sptoken)

        input_ids_a = np.asarray(sptoken + a_ids1 + b_ids1 + [tokenizer.eos_token_id], dtype=np.int32)
        attention_mask_a = np.asarray([1] * len(input_ids_a), dtype=np.int32)

        input_ids_b = np.asarray(sptoken + a_ids2 + b_ids2 + [tokenizer.eos_token_id], dtype=np.int32)
        attention_mask_b = np.asarray([1] * len(input_ids_b), dtype=np.int32)

        seqlen_a = len(input_ids_a)
        seqlen_b = len(input_ids_b)

        if seqlen_a == seqlen_b:
            if np.all(input_ids_a == input_ids_b):
                return None

        pos_a = len(a_ids1) + len(sptoken)
        pos_b = len(a_ids2) + len(sptoken)
        assert pos_a >= 0 and pos_a < max_seq_length - 1 and pos_b >= 0 and pos_b < max_seq_length - 1
        labels = np.asarray([-100] * pos_a + input_ids_a[pos_a:].tolist(), dtype=np.int64)
        labels2 = np.asarray([-100] * pos_b + input_ids_b[pos_b:].tolist(), dtype=np.int64)
        return {
            "input_ids": input_ids_a,
            "attention_mask": attention_mask_a,
            "labels": labels,
            # "seqlen": np.asarray(seqlen_a,dtype=np.int32),
            "input_ids2": input_ids_b,
            "attention_mask2": attention_mask_b,
            "labels2": labels2,
            # "seqlen2": np.asarray(seqlen_b, dtype=np.int32),
        }





class TokenIdsMakerForGLM(TokenIdsMaker):
    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,sptoken, max_seq_length: int,src_max_length,dst_max_length):
        prompt, chosen, rejected = pair_data


        a_ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_length,add_special_tokens=False)
        b_ids1 = tokenizer.encode(chosen, truncation=True, max_length=max_seq_length,add_special_tokens=False)
        b_ids2 = tokenizer.encode(rejected, truncation=True, max_length=max_seq_length,add_special_tokens=False)

        if src_max_length is not None and src_max_length > 0:
            a_ids = a_ids[:src_max_length]

        if dst_max_length is not None and dst_max_length > 0:
            b_ids1 = b_ids1[:dst_max_length]
            b_ids2 = b_ids2[:dst_max_length]

        a_ids1 = copy.deepcopy(a_ids)
        cls.trunction_ids(a_ids1,b_ids1,max_seq_length,sptoken)

        a_ids2 = copy.deepcopy(a_ids)
        cls.trunction_ids(a_ids2, b_ids2,max_seq_length,sptoken)


        input_ids_a = np.asarray( a_ids1 + sptoken + b_ids1 + [tokenizer.eos_token_id] ,dtype=np.int32)
        input_ids_b = np.asarray( a_ids2 + sptoken + b_ids2 + [tokenizer.eos_token_id] ,dtype=np.int32)


        seqlen_a = len(input_ids_a)
        seqlen_b = len(input_ids_b)

        if seqlen_a == seqlen_b:
            if np.all(input_ids_a == input_ids_b):
                return None

        pos_a = len(a_ids1)
        pos_b = len(a_ids2)
        assert pos_a >= 0 and pos_a < max_seq_length -1 and pos_b >= 0 and pos_b < max_seq_length -1
        labels = np.asarray([-100] * pos_a + input_ids_a[pos_a:].tolist(),dtype=np.int64)
        labels2 = np.asarray([-100] * pos_b + input_ids_b[pos_b:].tolist(),dtype=np.int64)

        ctxlen = pos_a + len(sptoken) - 1
        ctxlen2 = pos_b + len(sptoken) - 1

        ctxlen = np.asarray(ctxlen,dtype=np.int32)
        ctxlen2 = np.asarray(ctxlen2, dtype=np.int32)
        return {
            "input_ids": input_ids_a,
            "ctxlen": ctxlen,
            "labels": labels,
            # "seqlen": np.asarray(seqlen_a,dtype=np.int32),
            "input_ids2": input_ids_b,
            "ctxlen2": ctxlen2,
            "labels2": labels2,
            # "seqlen2": np.asarray(seqlen_b, dtype=np.int32),
        }


class TokenIdsMakerForGLM2(TokenIdsMaker):

    @classmethod
    def process(cls,pair_data,tokenizer: PreTrainedTokenizer,sptoken, max_seq_length: int,src_max_length,dst_max_length):
        prompt, chosen, rejected = pair_data


        a_ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_length,add_special_tokens=False)
        b_ids1 = tokenizer.encode(chosen, truncation=True, max_length=max_seq_length,add_special_tokens=False)
        b_ids2 = tokenizer.encode(rejected, truncation=True, max_length=max_seq_length,add_special_tokens=False)

        if src_max_length is not None and src_max_length > 0:
            a_ids = a_ids[:src_max_length]

        if dst_max_length is not None and dst_max_length > 0:
            b_ids1 = b_ids1[:dst_max_length]
            b_ids2 = b_ids2[:dst_max_length]

        a_ids1 = copy.deepcopy(a_ids)
        cls.trunction_ids(a_ids1,b_ids1,max_seq_length,sptoken)

        a_ids2 = copy.deepcopy(a_ids)
        cls.trunction_ids(a_ids2, b_ids2,max_seq_length,sptoken)


        input_ids_a = np.asarray( sptoken + a_ids1 + b_ids1 + [tokenizer.eos_token_id] ,dtype=np.int32)
        input_ids_b = np.asarray( sptoken + a_ids2 + b_ids2 + [tokenizer.eos_token_id] ,dtype=np.int32)


        seqlen_a = len(input_ids_a) + len(sptoken)
        seqlen_b = len(input_ids_b) + len(sptoken)

        if seqlen_a == seqlen_b:
            if np.all(input_ids_a == input_ids_b):
                return None

        pos_a = len(a_ids1)
        pos_b = len(a_ids2)
        assert pos_a >= 0 and pos_a < max_seq_length -1 and pos_b >= 0 and pos_b < max_seq_length -1
        labels = np.asarray([-100] * pos_a + input_ids_a[pos_a:].tolist(),dtype=np.int64)
        labels2 = np.asarray([-100] * pos_b + input_ids_b[pos_b:].tolist(),dtype=np.int64)


        return {
            "input_ids": input_ids_a,
            "labels": labels,
            # "seqlen": np.asarray(seqlen_a,dtype=np.int32),
            "input_ids2": input_ids_b,
            "labels2": labels2,
            # "seqlen2": np.asarray(seqlen_b, dtype=np.int32),
        }