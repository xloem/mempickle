import torch, numpy as np
import pickle
import struct
import mmap

gptj_24G = '/media/3/huggingface/transformers/026f12960ffab80e6f4f983cd8672c6f4579e3a12d469f0cad87973e86376f78.45ab80c413af010231b34f81ca5a6a2fe0739bbe17b4672a5110f70bb75d2555'
gpt2_1_5G = '/media/3/huggingface/transformers/6249eef5c8c1fcfccf9f36fc2e59301b109ac4036d8ebbee9c2b7f7e47f440bd.2538e2565f9e439a3668b981faf959c8b490b36dd631f3c4cd992519b2dd36f1'
gpt2_0_5G = '/home/ubuntu/.cache/huggingface/transformers/752929ace039baa8ef70fe21cdf9ab9445773d20e733cf693d667982e210837e.323c769945a351daa25546176f8208b3004b6f563438a7603e7932bae9025925'

class TensorMap:
    def __init__(self, filename):
        self.filename = filename
        self.version = 1
    @staticmethod
    def endian():
        return ['big', 'little'][np.array(1).tobytes()[0]]
    def exists(self):
        import os
        return os.path.exists(self.filename)
    def write(self, data):
        self.pagesize = mmap.PAGESIZE
        with open(self.filename, 'wb') as output:
            header = pickle.dumps((self.version, self.pagesize, len(data)))
            output.write(header)
            for name, tensor in data.items():
                flat_tensor = tensor.flatten()
                numpy = flat_tensor.numpy()
                numpy_dtype = numpy.dtype
                tensor_header = pickle.dumps((name, tensor.dtype, tuple(tensor.shape), numpy_dtype, len(numpy), tensor.requires_grad))
                output.write(tensor_header)
                pos = output.tell()
                if pos % self.pagesize != 0:
                    output.seek(pos - (pos % self.pagesize) + self.pagesize)
                numpy.tofile(output)
    def read(self):
        self.file = open(self.filename, 'rb')
        self.version, self.pagesize, data_len = pickle.load(self.file)
        assert self.pagesize % mmap.PAGESIZE == 0
        data = {}
        for idx in range(data_len):
            name, tensor_dtype, tensor_shape, numpy_dtype, numpy_len, requires_grad = pickle.load(self.file)
            pos = self.file.tell()
            if pos % self.pagesize != 0:
                pos += self.pagesize - (pos % self.pagesize)
            #buf = self.file.read(numpy_dtype.itemsize * numpy_len)
            bytelen = numpy_dtype.itemsize * numpy_len
            buf = mmap.mmap(self.file.fileno(), bytelen, access = mmap.ACCESS_READ, offset = pos)

            numpy = np.frombuffer(buf, numpy_dtype, count=numpy_len, offset=0)
            tensor = torch.from_numpy(numpy)
            tensor = tensor.unflatten(0, tensor_shape)
            tensor.requires_grad = requires_grad
            data[name] = tensor
            self.file.seek(pos + bytelen)
        return data

tensormap = TensorMap('test_output.tensormap')

if not tensormap.exists():
    data = torch.load(gpt2_0_5G)
    tensormap.write(data)

data = tensormap.read()
print(data.keys())
