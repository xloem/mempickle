#!/usr/bin/env python3

import transformers, torch, numpy as np
import os
import pickle
import struct
import mmap

PTMAP_WEIGHTS_NAME = transformers.file_utils.WEIGHTS_NAME.replace('.bin', '.ptmap')

class PyTorchMap:
    def __init__(self, filename = PTMAP_WEIGHTS_NAME):
        self.filename = filename
        self.version = 1

    @staticmethod
    def from_model(name_or_path, revision = None, mirror = None, cache_dir = None, force_download = False, proxies = None, resume_download = False, local_files_only = False, use_auth_token = None):
        if os.path.isdir(name_or_path):
            filename = os.path.join(name_or_path, PTMAP_WEIGHTS_NAME)
        else:
            filename = transformers.file_utils.hf_bucket_url(name_or_path, filename = PTMAP_WEIGHTS_NAME, revision = revision, mirror = mirror)
            filename = transformers.file_utils.cached_path(filename, cache_dir = cache_dir, force_download = force_download, proxies = proxies, resume_download = resume_download, local_files_only = local_files_only, use_auth_token = use_auth_token)
        return PyTorchMap(filename)

    endian = ['big', 'little'][np.array(1).tobytes()[0]]
    pagesize = mmap.PAGESIZE

    def exists(self):
        return os.path.exists(self.filename)

    def write(self, data):
        self.pagesize = mmap.PAGESIZE
        with open(self.filename, 'wb') as output:
            header = pickle.dumps((self.version, self.pagesize, len(data)))
            output.write(header)
            for name, tensor in data.items():
                flat_tensor = tensor.flatten()
                # numpy is only used because it seemed easier to quickly implement when writing this
                numpy = flat_tensor.numpy()
                numpy_dtype = numpy.dtype
                tensor_header = pickle.dumps((name, tensor.dtype, tuple(tensor.shape), numpy_dtype, len(numpy), tensor.requires_grad))
                output.write(tensor_header)
                pos = output.tell()
                if pos % self.pagesize != 0:
                    output.seek(pos - (pos % self.pagesize) + self.pagesize)
                numpy.tofile(output)

    def read(self, writable = False):
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

def pipeline(task = None, model = None, *params, framework = None, use_auth_token = None, revision = None, model_kwargs = {}, **kwparams):
    if 'state_dict' not in model_kwargs and framework in (None, 'pt'):
        if task is None and model is not None:
            task = transformers.pipelines.get_task(model, use_auth_token)
        targeted_task, task_options = transformers.pipelines.check_task(task)
        if model is None:
            model = transformers.pipelines.get_default_model(targeted_task, framework, task_options)
        model_kwargs['state_dict'] = PyTorchMap.from_model(model, revision = revision, use_auth_token = use_auth_token).read()
    return transformers.pipeline(task, model, *params, framework = framework, use_auth_token = use_auth_token, revision = revision, model_kwargs = model_kwargs, **kwparams)
    

if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='convert a .pt file or model to a .ptmap file')
    parser.add_argument('input_path', help='.pt file or model dir to convert')
    parser.add_argument('-o', '--output_filename', required=False, help='.ptmap file to output to')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        args.input_path = os.path.join(args.input_path, transformers.file_utils.WEIGHTS_NAME)

    if not args.output_filename:
        basename = args.input_path
        if basename.endswith('.pt'):
            basename = basename[:-len('.pt')]
        elif basename.endswith('.bin'):
            basename = basename[:-len('.bin')]
        args.output_filename = basename + '.ptmap'

    tensormap = PyTorchMap(args.output_filename)

    assert not tensormap.exists() or args.force

    print(f'Loading {args.input_path} ...', file=sys.stderr)
    torch_data = torch.load(args.input_path)

    print(f'Writing {args.output_filename} ...', file=sys.stderr)
    tensormap.write(torch_data)
    
