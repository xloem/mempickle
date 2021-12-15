
import torch
import pickle

gptj_24G = '/media/3/huggingface/transformers/026f12960ffab80e6f4f983cd8672c6f4579e3a12d469f0cad87973e86376f78.45ab80c413af010231b34f81ca5a6a2fe0739bbe17b4672a5110f70bb75d2555'
gpt2_1_5G = '/media/3/huggingface/transformers/6249eef5c8c1fcfccf9f36fc2e59301b109ac4036d8ebbee9c2b7f7e47f440bd.2538e2565f9e439a3668b981faf959c8b490b36dd631f3c4cd992519b2dd36f1'

with open(gpt2_1_5G, 'rb') as f:
    params = torch.load(f)#pickle.load(f)
