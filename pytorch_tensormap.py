#!/usr/bin/env python3

import transformers, torch, numpy as np
import json
import concurrent.futures
import mmap
import os
import pickle
import struct
import threading
import warnings

import patch_pytorch

WEIGHTS_NAME = transformers.file_utils.WEIGHTS_NAME
PTMAP_WEIGHTS_NAME = WEIGHTS_NAME.replace('.bin', '.ptmap')

class PyTorchMap:
    def __init__(self, filename = PTMAP_WEIGHTS_NAME):
        self.filename = filename
        self.version = 1

    endian = ['big', 'little'][np.array(1).tobytes()[0]]
    pagesize = mmap.PAGESIZE

    def write(self, data, verbose = True, pagesize = mmap.PAGESIZE):
        self.pagesize = pagesize
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

    def read(self, writeable = False, verbose = True, multi = False, add_prefix = '', track_forward_calls = True):
        self.file = open(self.filename, 'r+b' if writeable else 'rb')
        self.version, self.pagesize, data_len = pickle.load(self.file)
        assert self.pagesize % mmap.PAGESIZE == 0
        data = {}
        total_tensor_elements = 0
        enumeration = range(data_len)
        if verbose:
            import tqdm
            enumeration = tqdm.tqdm(enumeration, total = data_len, leave = False)
        access = mmap.ACCESS_DEFAULT if writeable else mmap.ACCESS_READ
        if not multi:
            buf = mmap.mmap(self.file.fileno(), 0, access = access, offset = 0)
        for idx in enumeration:
            name, tensor_dtype, tensor_shape, numpy_dtype, numpy_len, requires_grad = pickle.load(self.file)
            total_tensor_elements += numpy_len
            enumeration.set_description(name)
            pos = self.file.tell()
            if pos % self.pagesize != 0:
                pos += self.pagesize - (pos % self.pagesize)

            bytelen = numpy_dtype.itemsize * numpy_len
            if multi:
                buf = mmap.mmap(self.file.fileno(), bytelen, access = access, offset = pos)
                tensor_offset = 0
            else:
                tensor_offset = pos

            try:
                tensor = torch.frombuffer(buf, dtype = tensor_dtype, count = numpy_len, offset = tensor_offset, requires_grad = requires_grad)
            except AttributeError:
                numpy = np.frombuffer(buf, numpy_dtype, count = numpy_len, offset = tensor_offset)
                tensor = torch.from_numpy(numpy)
                tensor.requires_grad = requires_grad
            if not len(tensor_shape) and numpy_len == 1:
                tensor = tensor[0]
            else:
                tensor = tensor.unflatten(0, tensor_shape)
            data[add_prefix + name] = tensor
            self.file.seek(pos + bytelen)

        order_filename = self.filename + '.order.json'
        if os.path.exists(order_filename):
            try:
                with open(order_filename) as order_file:
                    order = json.load(order_file)
                    new_data = {}
                    for key in order:
                        new_data[key] = data[key]
                        del data[key]
                    new_data.update(data)
                    data = new_data
                    warnings.warn(f'Data presented reordered from file {order_filename}.')
                    warnings.warn('To use this order, resave the data and delete the order file.')
            except:
                pass

        #import pdb; pdb.set_trace()
        if track_forward_calls:
            self.track_forward_calls(data, total_tensor_elements, verbose = verbose)
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

    def track_forward_calls(self, state_dict, total_tensor_elements, verbose):

        out_of_order = False

        next_to_preload = concurrent.futures.Future()
        def preload_loop():
            nonlocal next_to_preload
            for tensor in range(len(state_dict)):
                if out_of_order:
                    print(f'thread waiting for {next_to_preload}')
                next_tensors = next_to_preload.result()
                if out_of_order:
                    print(f'thread got {next_to_preload}')
                next_to_preload = concurrent.futures.Future()
                if out_of_order:
                    print(f'thread will next wait for {next_to_preload}')
                for next_tensor in next_tensors:
                    next_tensor = next_tensor.flatten()
                    for idx in range(0, len(next_tensor), mmap.PAGESIZE):
                        next_tensor[idx]

        self.preload_thread = threading.Thread(target=preload_loop)
        self.preload_thread.start()

        tensor_ct = len(state_dict)
        key_tensor_by_stored_idx = [*state_dict.items()]
        stored_idx_key_tensor_by_data = {
            tensor.data_ptr(): (idx, key, tensor)
            for idx, (key, tensor)
            in enumerate(key_tensor_by_stored_idx)
        }
        next_stored_idx = 0
        order_matched = False
        module_tidx_key_tensor_by_read_idx = [] # add time to this.  make it a namedtuple.
        def track_forward(module, inputs):
            for tidx, (pname, parameter) in enumerate(module._parameters.items()):
                if parameter is None:
                    continue
                stored_idx_key_tensor = stored_idx_key_tensor_by_data.get(parameter.data_ptr())
                if stored_idx_key_tensor is not None:
                    stored_idx, key, tensor = stored_idx_key_tensor
                    print(key, tensor.shape, module.__class__.__name__, pname)

                    # set out_of_order flag
                    nonlocal out_of_order, next_stored_idx
                    if not out_of_order:
                        if stored_idx != next_stored_idx:
                            out_of_order = True
                    next_stored_idx = stored_idx + 1
                    if out_of_order:
                        print(f'main checking {next_to_preload}')
                    if not next_to_preload.done():
                        if out_of_order:
                            print('main appending result')
                        next_to_preload.set_result((key_tensor_by_stored_idx[next_stored_idx][1],))
                    else:
                        if out_of_order:
                            print('main skipping append due to last one not taken yet')

                    # store read order
                    if len(module_tidx_key_tensor_by_read_idx) == 0 or tensor is not module_tidx_key_tensor_by_read_idx[0][-1]:
                        module_tidx_key_tensor_by_read_idx.append((module, tidx, key, tensor))
                    else:
                        #import pdb; pdb.set_trace()
                        finished_tracking_calls()
                        break

        tracking_handle = torch.nn.modules.module.register_module_forward_pre_hook(track_forward)

        def finished_tracking_calls():
            tracking_handle.remove()
            if out_of_order:
                state_dict.clear()
                state_dict.update({
                    key : tensor
                    for module, tidx, key, tensor in module_tidx_key_tensor_by_read_idx
                })
                warnings.warn(f'tensors in {self.filename} are ordered differently from use.')
                warnings.warn(f'they have been sorted in the read state_dict and can be resaved.')
                order_filename = self.filename + '.order.json'
                with open(order_filename, 'wt') as order_file:
                    json.dump([*state_dict.keys()], order_file)
                warnings.warn(f'runtime key order has been written to {order_filename}')
                #import pdb; pdb.set_trace()
            last_module = None
            module_bin = []
            for (
                read_idx,
                (
                    (prev_module, prev_tidx, prev_key, prev_tensor),
                    (next_module, next_tidx, next_key, next_tensor)
                )
            ) in enumerate(zip(
                module_tidx_key_tensor_by_read_idx[:-1],
                module_tidx_key_tensor_by_read_idx[1:]
            )):
                #if last_module is None:
                #    last_module = prev_module
                #if prev_module is last_module:
                #    if next_module is last_module:
                #        # going thru start i guess
                #        pass
                #    elif next_module is not last_module:
                #        module_bin.append(next_tensor)
                #elif prev_module is not last_module:
                #    if next_module is prev_module:
                #        module_bin.append(next_tensor)
                #    elif next_module is not prev_module:
                #if next_module is not last_module:
                if prev_module in (last_module, next_module):
                    module_bin.append(next_tensor)
                else:
                    if last_module is not None:
                        register_module_next_tensor(read_idx - 1, last_module, module_bin)
                    module_bin.clear()
                    last_module = prev_module

        def register_module_next_tensor(read_idx, module, next_tensors):
            # the tensors are smaller than ram individually; it is fine to preload them together
            def prepare_next_tensor(module, input):
                if out_of_order:
                    print(f'main2 checking {next_to_preload}')
                if not next_to_preload.done():
                    if out_of_order:
                        print('main2 setting result')
                    next_to_preload.set_result(next_tensors)
                else:
                    if out_of_order:
                        print('main2 skipping due to last result not taken')
                if not self.preload_thread.is_alive():
                    self.preload_thread = threading.Thread(target=preload_loop)
                    self.preload_thread.start()

            module.register_forward_pre_hook(prepare_next_tensor)

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
            #import pdb; pdb.set_trace()
            try:
                result = PyTorchMap._cache.get(fn, None)
                if result is None:
                    result = PyTorchMap(fn).read(**self.read_kwparams)
                    PyTorchMap._cache[fn] = result
                return result
            except:
                return self._torch_load(fn, *params, **kwparams)
        torch.load = torch_load_wrapper
        
        self._torch_save = torch.save
        def torch_save_wrapper(obj, fn, *params, **kwparams):
            if fn in PyTorchMap._cache or fn.endswith('.ptmap'):
                if fn in PyTorchMap._cache:
                    new_obj = {}
                    for key in PyTorchMap._cache[fn].keys():
                        if key in obj:
                            new_obj[key] = obj[key]
                            del obj[key]
                    new_obj.update(obj)
                    obj.update(new_obj)
                    obj = new_obj
                PyTorchMap(fn).write(obj, verbose = self.read_kwparams.get('verbose', True))
            else:
                self._torch_save(obj, fn, *params, **kwparams)
        torch.save = torch_save_wrapper

        self._transformers_pipeline = transformers.pipeline
        def pipeline_wrapper(*params, model_kwargs = None, **kwparams):
            if model_kwargs is None:
                model_kwargs = {}
            model_kwargs['low_cpu_mem_usage'] = True
            return self._transformers_pipeline(*params, model_kwargs = model_kwargs, **kwparams)
        transformers.pipeline = pipeline_wrapper
        def Linear_wrapper(in_features, out_features, bias = True, device = None, dtype = None):
            return torch.nn.LazyLinear(out_features, bias, device, dtype)
        self._torch_nn_linear = torch.nn.Linear
        torch.nn.Linear = Linear_wrapper

    def __exit__(self, *params, **kwparams):
        torch.nn.Linear = self._torch_nn_linear
        transformers.pipeline= self._transformers_pipeline
        torch.save = self._torch_save
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
    
