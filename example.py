#!/usr/bin/env python3

import torch, transformers

import pytorch_tensormap

### load pipeline with ptmap files ###
# unfortunately this context manager presently hacks the inside of transformers and may fail on other versions.
with pytorch_tensormap.Ctx(offline = False, writeable = True):
    gpt2 = transformers.pipeline('text-generation', model = 'baffo32/gpt2-ptmap')


### demo sugar ###

prompt = input('Enter a prompt: ').rstrip()

all_token_ids = torch.arange(0, gpt2.tokenizer.vocab_size, dtype=torch.long)
all_token_ids_except_eos = torch.cat((
    all_token_ids[:gpt2.tokenizer.eos_token_id],
    all_token_ids[gpt2.tokenizer.eos_token_id + 1:]
))

import curses
curses.setupterm()
display_width = curses.tigetnum('cols')
display_height = curses.tigetnum('lines')
num_sequences = display_height - 2
buffers = [' ' * display_width] * num_sequences
last_ends = [0] * num_sequences
cur_line = 0
ticup = curses.tigetstr('cup')

for row in range(num_sequences):
    print()

def token_update(batch_id, input_ids):
    global buffers, last_ends, cur_line

    tokens = gpt2.tokenizer.decode(input_ids[last_ends[batch_id]:])
    tokens = tokens.replace('\t','    ').replace('\n', '  ')
    last_ends[batch_id] = len(input_ids)
    
    buffer = buffers[batch_id]
    buffer = buffer[len(tokens):] + tokens
    buffers[batch_id] = buffer

    print(curses.tparm(ticup, batch_id + 2, 0).decode() + buffer, end='', flush = (batch_id + 1 == num_sequences))

    return all_token_ids_except_eos

gpt2(prompt, prefix_allowed_tokens_fn = token_update, pad_token_id = gpt2.tokenizer.eos_token_id, max_length = 1024, num_return_sequences = num_sequences)

print()
