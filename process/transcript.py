from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from functools import lru_cache
from typing import Callable, Union, Literal
import json
import os
import re

import nltk
import numpy as np
import torch
from transformers import AutoTokenizer, pipeline

from .static import transcript_dir, keyword_file

# Match_Func = Callable[[Union[str, list[str]]], list[tuple[int, int]]]
_MODEL_NAMES = {
    'clip': 'openai/clip-vit-base-patch16',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-base-v2',
    'xlm-roberta': 'xlm-roberta-base',
    'xlnet': 'xlnet-base-cased',
    'electra': 'google/electra-base-discriminator',
    'bart': 'facebook/bart-base',
    't5': 't5-small',
}


def get_plural_words(word: str):
    """
    Get *likely* plural words of a word
    Args:
        word: a word
    Returns:
        A list of words that are likely to be plural form of the word
    """
    if word.endswith('y') and word[-2] not in 'aeiou':
        return word[:-1] + 'ies',
    elif word.endswith('f'):
        return word[:-1] + 'ves',
    elif word.endswith('fe'):
        return word[:-2] + 'ves',
    elif word.endswith('o'):
        return word + 'es', word + 's'
    elif word.endswith('sis'):
        return word[:-2] + 'ses',
    elif word.endswith('sh') or word.endswith('ch') or word.endswith('s') or word.endswith('x') or word.endswith(
            'z'):
        return word + 'es',
    else:
        return word + 's',


@lru_cache(None)
def get_keyword_set():
    """
    Get keyword set (LRU-cached) \n
    All kinds of potential forms (plural, consecutive) are included

    Returns:
        A set of keywords
    """
    with open(keyword_file, 'r') as f:
        keywords = json.load(f)
    keyword_set = set()
    for keyword in keywords:
        keyword_set.add(keyword)
        for word in get_plural_words(keyword):
            keyword_set.add(word)
        keyword_set.add(keyword.replace('_', ''))
    return keyword_set


@lru_cache(None)
def get_tokenizer(model_name: str, use_fast: bool = True):
    """
    Get a tokenizer (LRU-cached)
    Args:
        model_name: name of the model
        use_fast: whether to use fast tokenizer

    Returns:
        A tokenizer
    """
    if model_name in _MODEL_NAMES:
        model_name = _MODEL_NAMES[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    return tokenizer


@lru_cache(None)
def get_summarizer(model_name: str, device: int):
    """
    Get a summarizer (LRU-cached)
    Args:
        model_name:  name of the model
        device:  device id

    Returns:
        A summarizer
    """
    return pipeline('summarization', model=model_name, device=device, truncation=True)


def get_transcript(vid: str):
    """
    Get transcript with video id
    Args:
        vid: video id

    Returns:
        A list of dict, each dict contains 'text','start','duration'
    """
    transcript_path = os.path.join(transcript_dir, vid + '.srt')
    assert os.path.isfile(transcript_path), f'Transcript {transcript_path} does not exist'
    with open(transcript_path, 'r') as f:
        srt = json.load(f)
    return srt


def get_transcript_new(clip: str):
    """
    Get transcript with clip id
    Args:
        clip: clip id

    Returns:
        A list of dict, each dict contains 'transcript','keyword','pos'
    """
    transcript_path = os.path.join(clip, 'log.json')
    assert os.path.isfile(transcript_path), f'Transcript {transcript_path} does not exist'
    with open(transcript_path, 'r') as f:
        srt = json.load(f)
    return srt


def get_format_transcript(vid: str, return_dict: bool = False):
    """
    Get a transcript with video id and format it into words sequence and relative timestamps
    Args:
        vid: video id
        return_dict: whether to return a dict
    Returns:
        (words,timestamps) when return_dict is False {'words':words,'timestamps':timestamps} when return_dict is True
    """
    srt = get_transcript(vid)
    texts = [srt_clip['text'] for srt_clip in srt]
    starts = [srt_clip['start'] for srt_clip in srt]
    durations = [srt_clip['duration'] for srt_clip in srt]
    for i in range(len(starts) - 1):
        durations[i] = min(starts[i + 1] - starts[i], durations[i])

    words = []
    timestamps = []
    for i, text in enumerate(texts):
        clip_words = nltk.word_tokenize(text.lower())
        clip_length = len(clip_words)
        clip_timestamps = list(np.linspace(starts[i], starts[i] + durations[i], clip_length))
        words.extend(clip_words)
        timestamps.extend(clip_timestamps)

    return {'words': words, 'timestamps': timestamps} if return_dict else (words, timestamps)


def keyword_match(sentence: str | list[str], find_all: bool = False):
    """
    Match keywords in a sequence of words
    Args:
        sentence: a sequence of words or a sentence
        find_all: whether to find all matches
    Returns:
        List of (index,subsequence_length) when find_all is True
        Boolean to illustrate whether keyword is in the sequence when find_all is False
    """

    if isinstance(sentence, str):
        words = nltk.word_tokenize(sentence)
    else:
        words = sentence
    assert len(words) > 0, 'Invalid sentence : No words.'

    keywords = get_keyword_set()
    ans = []
    i = 0

    while i < len(words):
        if not find_all and len(ans) > 0:
            return True

        word = words[i]
        if i < len(words) - 1:
            next_word = words[i + 1]
            if word + '_' + next_word in keywords:
                ans.append((i, 2))
                i += 2
                continue

        if word in keywords:
            ans.append((i, 1))
        i += 1

    return ans if find_all else False


def action_match(sentence: str | list[str]):
    """
    Find verbs in a sequence of words

    Args:
        sentence: a sequence of words or a sentence

    Returns:
        List of (index,subsequence_length)
    """
    if isinstance(sentence, str):
        words = nltk.word_tokenize(sentence)
    else:
        words = sentence

    ans = []
    pos_tag = nltk.pos_tag(words)
    for i, (word, tag) in enumerate(pos_tag):
        if tag.startswith('VB'):
            ans.append((i, 1))

    return ans


def transcript_to_clips(vid: str,
                        target_length: int,
                        record_pos: bool = True,
                        overlap: bool = False,
                        stride: int | None = None, ):
    """
    Turn the transcript of given video id into clips containing keywords

    Args:
        vid: video id
        target_length: target length of each clip
        record_pos: whether to record relative position of keywords in each clip
        overlap: whether to clip with overlap
        stride: minimum gap between the starts of two clips when overlap is True

    Returns:
        A list of dict, each dict contains words, timestamps and keywords positions (if record_pos is True) of a clip
    """
    if overlap:
        assert stride is not None, 'Stride must be specified when overlap is True'

    transcript = get_format_transcript(vid, return_dict=True)

    words = transcript['words']
    timestamps = transcript['timestamps']
    assert len(words) == len(timestamps), 'Invalid transcript : words and timestamps mismatch.'

    keywords = keyword_match(words, find_all=True)

    ans = []
    ptr, nxt_ptr = 0, None

    while ptr < len(keywords):

        index_start = keywords[ptr][0]

        keywords_index = []
        while ptr < len(keywords):
            tmp_index = keywords[ptr][0]
            gap = tmp_index - index_start
            if nxt_ptr is None and overlap and gap >= stride:
                nxt_ptr = ptr
            if gap >= target_length:
                break
            if record_pos:
                keywords_index.extend([tmp_index + i for i in range(keywords[ptr][1])])
            ptr += 1

        index_end = keywords[ptr - 1][0] + keywords[ptr - 1][1]
        clip_start = max(0, (index_start + index_end) // 2 - target_length // 2)
        clip_end = min(clip_start + target_length, len(words))

        clip_words = words[clip_start:clip_end]
        clip_timestamps = timestamps[clip_start:clip_end]
        if record_pos:
            clip_keywords = [keyword_index - clip_start for keyword_index in keywords_index]
            ans.append({'words': clip_words, 'timestamps': clip_timestamps, 'keywords': clip_keywords})
        else:
            ans.append({'words': clip_words, 'timestamps': clip_timestamps})
        if overlap:
            ptr = nxt_ptr
            nxt_ptr = None
    return ans


def clip_summary(clip: list[str] | str, max_length: int, min_length: int, summarizer: str = 'bert', device: int = 0):
    """
    Summarize the clips into a clip
    Args:
        clip: a list of clips or a clip
        max_length: maximum length of the summary
        min_length: minimum length of the summary
        summarizer: summarizer to use
        device: device to use

    Returns:
        Shorter clips summarizing the clips
    """
    if isinstance(clip, str):
        clip = [clip]
    assert len(clip) > 0, 'Invalid clip : No clips.'

    model = get_summarizer(summarizer, device)
    summaries = model(clip, max_length=max_length, min_length=min_length, do_sample=False)
    return [summary['summary_text'] for summary in summaries]


def transcript_to_clips_new(clip: str,
                            target_length: int,
                            record_pos: bool = True,
                            overlap: bool = False,
                            stride: int | None = None, ):
    """
    Turn the transcript of given video id into clips containing keywords

    Args:
        clip: clip id
        target_length: target length of each clip
        record_pos: whether to record relative position of keywords in each clip
        overlap: whether to clip with overlap
        stride: minimum gap between the starts of two clips when overlap is True

    Returns:
        A list of dict, each dict contains words, timestamps and keywords positions (if record_pos is True) of a clip
    """

    srt = get_transcript_new(clip)

    clip_word = nltk.word_tokenize(srt['transcript'].lower())

    return {'words': clip_word, 'pos': srt['pos'], 'keyword': srt['keyword']} if record_pos \
        else {'words': clip_word, 'pos': srt['pos']}


def clip_to_tokens(clip_words: list[str] | str,
                   max_length: int,
                   tokenizer_name: str,
                   batch: bool = False,
                   keywords: list[int] | None = None,
                   mark_methods: str | list[str] | None = None,
                   padding: Literal["max_length", "longest", "none"] = "max_length",
                   ):
    """
    Tokenize *a* clip with selected tokenizer, and optionally mark several words

    Args:
        clip_words: a list of words in the clip
        batch: whether to tokenize a batch of clips, invalid when keywords or mark_func is not None
        keywords: a list of keyword positions in the clip
        mark_methods: a method to mark words, or a list of methods
        tokenizer_name: name of tokenizer
        max_length: max length of tokens
        padding: padding method, "max_length", "longest" or "none"

    Returns:
        A batch of tokens [Torch.LongTensor]
        Or a dict containing tokens[Torch.LongTensor], keyword_mask[Torch.LongTensor] and mark_mask[Torch.LongTensor]
    """

    if isinstance(clip_words, str):
        clip_words = nltk.word_tokenize(clip_words)

    tokenizer = get_tokenizer(tokenizer_name)
    if tokenizer_name == "clip":
        assert max_length <= 77, 'Invalid max_length : max_length should be less than 77 for clip tokenizer.'
        begin_token_id = tokenizer.bos_token_id
        end_token_id = tokenizer.eos_token_id
        pad_token_id = 0
    else:
        raise NotImplementedError
    words_tokens = tokenizer(clip_words, add_special_tokens=False)['input_ids']

    # batch mode
    if batch:
        assert keywords is None and mark_methods is None, \
            'Invalid arguments : keywords and mark_func should be None when batch is True.'
        if padding == "max_length":
            length = max_length
        elif padding == "longest":
            length = max([len(tokens) for tokens in words_tokens])
        else:
            raise NotImplementedError
        batch_tokens = torch.ones((len(clip_words), length), dtype=torch.long) * pad_token_id
        batch_tokens[:, 0] = begin_token_id
        for i, tokens in enumerate(words_tokens):
            tokens = tokens[:length - 2]
            batch_tokens[i, 1:len(tokens) + 1] = torch.LongTensor(tokens)
            batch_tokens[i, len(tokens) + 1] = end_token_id
        return batch_tokens

    # single clip mode
    # record token index of each word
    length = 1
    end_word_id = len(clip_words) - 1
    tokens_index = []
    clip_tokens = [begin_token_id]
    for i, word_tokens in enumerate(words_tokens):
        extend_length = min(len(word_tokens), max_length - length - 1)
        clip_tokens.extend(word_tokens[:extend_length])
        tokens_index.append([length + j for j in range(extend_length)])
        length += extend_length
        if length >= max_length - 1:
            end_word_id = i
            break
    clip_tokens.append(end_token_id)
    assert len(clip_tokens) == length + 1, 'Invalid clip_tokens : length mismatch.'  # Todo: remove this line

    # padding
    if padding == "max_length":
        padded_tokens = torch.ones(max_length, dtype=torch.long) * pad_token_id
        padded_tokens[:length + 1] = torch.LongTensor(clip_tokens)
        clip_tokens = padded_tokens
    elif padding == "none":
        clip_tokens = torch.LongTensor(clip_tokens)
    else:
        raise NotImplementedError

    # record keywords mask
    def get_flatten_index(words_id: list[int]):
        flatten_index = []
        for word_id in words_id:
            if word_id > end_word_id:
                break
            flatten_index.extend(tokens_index[word_id])
        return flatten_index

    if keywords is not None:
        keyword_mask = torch.zeros(len(clip_tokens), dtype=torch.long)
        keyword_mask[get_flatten_index(keywords)] = 1
    else:
        keyword_mask = None

    # record mark mask
    if mark_methods is None:
        mark_methods = []
    elif not isinstance(mark_methods, list):
        mark_methods = [mark_methods]

    mark_func = []
    for m in mark_methods:
        if m == 'action':
            mark_func.append(action_match)
        elif m == 'keyword':
            mark_func.append(lambda x: keyword_match(x, True))
        else:
            raise NotImplementedError

    mark_masks = []
    for func in mark_func:
        mark_mask = torch.zeros(len(clip_tokens), dtype=torch.long)
        mark_id = []
        for index, length in func(clip_words):
            mark_id.extend([index + i for i in range(length)])
        mark_mask[get_flatten_index(mark_id)] = 1
        mark_masks.append(mark_mask)

    return {'tokens': clip_tokens, 'keyword_mask': keyword_mask, 'mark_masks': mark_masks}


@torch.no_grad()
def tokens_to_hidden_states(tokens: torch.LongTensor,
                            language_model: torch.nn.Module,
                            device: str | torch.device,
                            mask_list: list[torch.LongTensor] | None = None,
                            full_output: bool = False,
                            padding: Literal["max_length", "longest", "none"] = "max_length",
                            max_length: int | None = None,
                            ):
    """
    Get hidden states of tokens from language model and extract the corresponding hidden states

    Args:
        tokens: tokens
        language_model: language model (function get_hidden_states will be used to get hidden states)
        device: device
        mask_list: list of masks to extract hidden states
        full_output: whether to return full hidden states
        padding: whether to pad tokens
        max_length: max length of tokens when padding

    Returns:
        hidden states of tokens OR a list of hidden states of tokens when mask_list is not None
    """

    assert len(tokens.shape) == 2, 'Invalid tokens : tokens should be a batch of tokens.'

    if padding == 'max_length':
        assert max_length is not None, 'Invalid arguments : max_length should not be None when padding is True.'

    language_model.eval()
    language_model.to(device)
    tokens = tokens.to(device)
    with torch.no_grad():
        hidden_states = language_model.get_hidden_state(tokens, full=True)

    B, L, D = hidden_states.shape

    if full_output:
        sentence_hidden_states = hidden_states.cpu().numpy()
    else:
        end_pos = tokens.argmax(dim=-1)
        sentence_hidden_states = hidden_states[torch.arange(len(end_pos)), end_pos].cpu().numpy()

    mark_hidden_states_list = []
    if mask_list is not None:
        for mask in mask_list:
            assert len(mask.shape) == 2, 'Invalid mask : mask should be a batch of masks.'
            mask = mask.to(device)
            mask_hidden_states = hidden_states[mask == 1]
            mask_count = mask.sum(-1)

            if padding == 'max_length':
                mark_length = max_length
                mark_hidden_states = torch.zeros((B, mark_length, D), dtype=torch.float32)
            elif padding == 'longest':
                mark_length = mask_count.max()
                mark_hidden_states = torch.zeros((B, mark_length, D), dtype=torch.float32)
            else:
                mark_length = None
                mark_hidden_states = []

            last = 0
            for i, count in enumerate(mask_count):
                if padding == 'none':
                    mark_hidden_states.append(mask_hidden_states[last:last + count].cpu().numpy())
                else:
                    hidden_length = min(count, mark_length)
                    mark_hidden_states[i, :count] = mask_hidden_states[last:last + hidden_length]
                last += count

            mark_hidden_states_list.append(mark_hidden_states if padding == 'none'
                                           else mark_hidden_states.cpu().numpy())

    return sentence_hidden_states, mark_hidden_states_list


def tokens_to_words(tokens: torch.LongTensor,
                    tokenizer_name: str,
                    ):
    """
    Get words from tokens

    Args:
        tokens: tokens
        tokenizer_name: name of tokenizer

    Returns:
        words
    """
    assert tokenizer_name == 'clip', 'Invalid arguments : tokenizer_name should be clip.'
    tokenizer = get_tokenizer(tokenizer_name)
    if len(tokens.shape) == 1:
        tokens = tokens.unsqueeze(0)
    words = []

    for sentence_tokens in tokens:
        sentence_end = sentence_tokens.argmax(dim=-1)
        sentence_tokens = sentence_tokens[:sentence_end]
        sentence_words = tokenizer.decode(sentence_tokens)
        words.append(sentence_words)

    return words


def extract_gpt(text):
    def extract_quoted_string(text):
        pattern = r'"(.*)"'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None

    def extract_content_1(text):
        pattern = r"Here's [\w\s]+:"
        matches = re.findall(pattern, text)
        if not matches:
            return -1
        last_match = matches[-1]
        match_index = text.rfind(last_match)
        match_index += len(last_match) - 1
        start_index, end_index = -1, -1
        start_index = text.find('"', match_index) + 1
        end_index = text.find('"', start_index)
        if start_index != -1 and end_index != -1 and start_index != end_index:
            return text[start_index:end_index]
        return -1

    def extract_content_2(text):
        pattern = r'(?s):[\r\n\s]*"([\w\s\S]*?)"'
        matches = re.findall(pattern, text)
        # print(matches)
        if not matches:
            return -1
        last_match = matches[-1]
        return last_match
        # print(last_match)
        # # quoted_text = extract_quoted_string(last_match)
        # print(quoted_text)
        # if quoted_text:
        #     return quoted_text
        # return -1

    def has_two_quotes(text):
        pattern = r'^"(.*)"$'
        match = re.search(pattern, text)
        return match is not None

    def should_delete(text):
        if "it is difficult" in text or "I'm sorry" in text or "is unclear" in text:
            return True
        else:
            return False
    text = text.strip()
    extracted_text = ''
    if has_two_quotes(text):
        extracted_text = text[1:-1]
    elif should_delete(text):
        return None
    elif extract_content_1(text) != -1:
        extracted_text = extract_content_1(text)
    elif extract_content_2(text) != -1:
        extracted_text = extract_content_2(text)
    else:
        extracted_text = text
    return extracted_text