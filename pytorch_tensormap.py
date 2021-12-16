#!/usr/bin/env python3

import transformers, torch, numpy as np
import os
import pickle
import struct
import mmap

WEIGHTS_NAME = transformers.file_utils.WEIGHTS_NAME
PTMAP_WEIGHTS_NAME = WEIGHTS_NAME.replace('.bin', '.ptmap')

class PyTorchMap:
    def __init__(self, filename = PTMAP_WEIGHTS_NAME):
        self.filename = filename
        self.version = 1

    endian = ['big', 'little'][np.array(1).tobytes()[0]]
    pagesize = mmap.PAGESIZE

    def write(self, data, verbose = True):
        self.pagesize = mmap.PAGESIZE
        with open(self.filename, 'wb') as output:
            header = pickle.dumps((self.version, self.pagesize, len(data)))
            output.write(header)
            enumerated_items = enumerate(data.items())
            if verbose:
                import tqdm
                enumerated_items = tqdm.tqdm(enumerated_items, total = len(data), leave = False)
            for idx, (name, tensor) in enumerated_items:
                if verbose:
                    enumerated_items.set_description(name)
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

    def read(self, writeable = False, verbose = True):
        self.file = open(self.filename, 'r+b' if writeable else 'rb')
        self.version, self.pagesize, data_len = pickle.load(self.file)
        assert self.pagesize % mmap.PAGESIZE == 0
        data = {}
        enumeration = range(data_len)
        if verbose:
            import tqdm
            enumeration = tqdm.tqdm(enumeration, total = data_len, leave = False)
        for idx in enumeration:
            name, tensor_dtype, tensor_shape, numpy_dtype, numpy_len, requires_grad = pickle.load(self.file)
            enumeration.set_description(name)
            pos = self.file.tell()
            if pos % self.pagesize != 0:
                pos += self.pagesize - (pos % self.pagesize)
            #buf = self.file.read(numpy_dtype.itemsize * numpy_len)
            bytelen = numpy_dtype.itemsize * numpy_len
            buf = mmap.mmap(self.file.fileno(), bytelen, access = mmap.ACCESS_DEFAULT if writeable else mmap.ACCESS_READ, offset = pos)

            numpy = np.frombuffer(buf, numpy_dtype, count=numpy_len, offset=0)
            tensor = torch.from_numpy(numpy)
            tensor = tensor.unflatten(0, tensor_shape)
            tensor.requires_grad = requires_grad
            data['transformer.' + name] = tensor
            self.file.seek(pos + bytelen)
        return data

    def exists(self):
        return os.path.exists(self.filename)

    @staticmethod
    def from_model(name_or_path, revision = None, mirror = None, cache_dir = None, force_download = False, proxies = None, resume_download = False, local_files_only = False, use_auth_token = None):
        if os.path.isdir(name_or_path):
            filename = os.path.join(name_or_path, PTMAP_WEIGHTS_NAME)
        else:
            filename = transformers.file_utils.hf_bucket_url(name_or_path, filename = PTMAP_WEIGHTS_NAME, revision = revision, mirror = mirror)
            filename = transformers.file_utils.cached_path(filename, cache_dir = cache_dir, force_download = force_download, proxies = proxies, resume_download = resume_download, local_files_only = local_files_only, use_auth_token = use_auth_token)
        return PyTorchMap(filename)

class Ctx:
    def __init__(self, offline : bool = None, **read_kwparams):
        self.offline = offline
        self.read_kwparams = read_kwparams
    def __enter__(self):
        self._transformers_offline = transformers.file_utils._is_offline_mode
        if self.offline is not None:
            transformers.file_utils._is_offline_mode = self.offline
        transformers.file_utils.WEIGHTS_NAME = PTMAP_WEIGHTS_NAME
        PyTorchMap._cache = {}
        self._torch_load = torch.load
        def torch_load_wrapper(fn, *params, **kwparams):
            try:
                result = PyTorchMap._cache.get(fn, None)
                if result is None:
                    result = PyTorchMap(fn).read(**self.read_kwparams)
                    PyTorchMap._cache[fn] = result
                return result
            except:
                return _torch_load(fn, *params, **kwparams)
        torch.load = torch_load_wrapper
        self._pipeline = transformers.pipeline
        def pipeline_wrapper(*params, model_kwargs = None, **kwparams):
            if model_kwargs is None:
                model_kwargs = {}
            model_kwargs['low_cpu_mem_usage'] = True
            return self._pipeline(*params, model_kwargs = model_kwargs, **kwparams)
        transformers.pipeline = pipeline_wrapper

    def __exit__(self, *params, **kwparams):
        transformers.pipeline= self._pipeline
        torch.load = self._torch_load
        transformers.file_utils.WEIGHTS_NAME = WEIGHTS_NAME
        transformers.file_utils._is_offline_mode = self._transformers_offline


if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='convert a .pt file or model to a .ptmap file')
    parser.add_argument('input_path', help='.pt file or model dir to convert')
    parser.add_argument('-o', '--output_filename', required=False, help='.ptmap file to output to')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    parser.add_argument('-q', '--quiet', '-s', '--silent', action='store_true', help='disable saving progress meter')
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

    if not args.quiet:
        print(f'Loading {args.input_path} ...', file=sys.stderr)
    torch_data = torch.load(args.input_path)

    if not args.quiet:
        print(f'Writing {args.output_filename} ...', file=sys.stderr)
    tensormap.write(torch_data, verbose=not args.quiet)
    
