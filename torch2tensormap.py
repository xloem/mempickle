import torch
import pickle
import struct

gptj_24G = '/media/3/huggingface/transformers/026f12960ffab80e6f4f983cd8672c6f4579e3a12d469f0cad87973e86376f78.45ab80c413af010231b34f81ca5a6a2fe0739bbe17b4672a5110f70bb75d2555'
gpt2_1_5G = '/media/3/huggingface/transformers/6249eef5c8c1fcfccf9f36fc2e59301b109ac4036d8ebbee9c2b7f7e47f440bd.2538e2565f9e439a3668b981faf959c8b490b36dd631f3c4cd992519b2dd36f1'

data = torch.load(gpt2_1_5G)
outputfn = 'test_output.tensormap'

class TensorMap:
    def __init__(self, filename):
        self.filename = filename
        self.version = 1
    def write(self, data):
        with open(self.filename, 'wb') as output:
            header = pickle.dumps((self.version, len(data))
            output.write(header)
            for name, tensor in data.items():
                flat_tensor = tensor.flatten()
                tensor_header = pickle.dumps((name, tensor.dtype, tuple(tensor.shape), len(flat_tensor), tensor.requires_grad))
                output.write(tensor_header)
