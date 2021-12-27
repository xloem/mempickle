#!/usr/bin/env python3

import transformers, torch, numpy as np
import json
import concurrent.futures
import mmap
import os
import pickle
import queue
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
        self._preload_loops = {}

    endian = ['big', 'little'][np.array(1).tobytes()[0]]
    pagesize = mmap.PAGESIZE

    def write(self, data, verbose = True, pagesize = mmap.PAGESIZE, enforce_aligned = False, remove_prefix = '', dtype = None):
        if enforce_aligned:
            assert pagesize % mmap.PAGESIZE == 0
        self.pagesize = pagesize
        if dtype:
            if type(dtype) is str:
                dtype = getattr(torch, dtype)
        if len(remove_prefix) and remove_prefix[-1] != '.':
            remove_prefix += '.'
        with open(self.filename, 'wb') as output:
            header = pickle.dumps((self.version, self.pagesize, len(data)))
            output.write(header)
            enumerated_items = enumerate(data.items())
            if verbose:
                import tqdm
                enumerated_items = tqdm.tqdm(enumerated_items, total = len(data), leave = False, unit = 'wt')
            for idx, (name, tensor) in enumerated_items:
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                if verbose:
                    enumerated_items.set_description(name)
                flat_tensor = tensor.flatten()
                if dtype and dtype is not flat_tensor.dtype:
                    flat_tensor = flat_tensor.to(dtype)
                try:
                    # numpy is only used because it seemed easier to quickly implement when writing this
                    numpy = flat_tensor.numpy()
                    numpy_dtype = numpy.dtype
                except TypeError: # numpy doesn't support all pytorch types
                    numpy = None
                    numpy_dtype = None
                tensor_header = pickle.dumps((name, flat_tensor.dtype, tuple(tensor.shape), numpy_dtype, len(flat_tensor), tensor.requires_grad))
                output.write(tensor_header)
                pos = output.tell()
                if pos % self.pagesize != 0:
                    output.seek(pos - (pos % self.pagesize) + self.pagesize)
                if numpy is not None:
                    numpy.tofile(output)
                else:
                    import ctypes
                    bytect = len(flat_tensor) * flat_tensor.element_size()
                    carray = (ctypes.c_char * bytect).from_address(flat_tensor.data_ptr())
                    output.write(carray)
                    #array = ctypes.cast(flat_tensor.data_ptr(), ctypes.POINTER(ctypes.c_char))
                    #output.write(array[:len(flat_tensor)*flat_tensor.element_size()])

    def read(self, writeable = False, verbose = True, multi = False, add_prefix = '', track_forward_calls = False, tracking_preloads_tensors = True, enforce_aligned = False, retain_unused = False):
        self.file = open(self.filename, 'r+b' if writeable else 'rb')
        self.version, self.pagesize, data_len = pickle.load(self.file)
        if multi or enforce_aligned:
            assert self.pagesize % mmap.PAGESIZE == 0
        if len(add_prefix) and add_prefix[-1] != '.':
            add_prefix += '.'
        data = {}
        total_tensor_bytes = 0
        enumeration = range(data_len)
        if verbose:
            import tqdm
            enumeration = tqdm.tqdm(enumeration, total = data_len, leave = False, unit = 'wt')
        access = mmap.ACCESS_DEFAULT if writeable else mmap.ACCESS_READ
        if not multi:
            buf = mmap.mmap(self.file.fileno(), 0, access = access, offset = 0)
        for idx in enumeration:
            name, tensor_dtype, tensor_shape, numpy_dtype, numpy_len, requires_grad = pickle.load(self.file)
            enumeration.set_description(name)
            pos = self.file.tell()
            if pos % self.pagesize != 0:
                pos += self.pagesize - (pos % self.pagesize)

            bytelen = torch.tensor(0, dtype=tensor_dtype).element_size() * numpy_len
            total_tensor_bytes += bytelen
            if multi:
                buf = mmap.mmap(self.file.fileno(), bytelen, access = access, offset = pos)
                tensor_offset = 0
            else:
                tensor_offset = pos

            try:
                tensor = torch.frombuffer( # readonly warning; use read(writeable = True) to write
                        buf, dtype = tensor_dtype, count = numpy_len, offset = tensor_offset, requires_grad = requires_grad)
            except AttributeError:
                warnings.warn('torch does not support buffer loading')
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
                        if not key.startswith(add_prefix):
                            key = add_prefix + key
                        new_data[key] = data[key]
                        del data[key]
                    if retain_unused:
                        new_data.update(data)
                    data = new_data
                    warnings.warn(
                        f'\nData presented reordered from file {order_filename}.'
                        '\nTo use this order, resave the data and delete the order file.'
                    )
            except:
                pass
        if track_forward_calls:
            self.track_forward_calls(data, total_tensor_bytes, add_prefix = add_prefix, verbose = verbose, preload = tracking_preloads_tensors, retain_unused = retain_unused)
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

    def cancel_preload(self, state_dict = None):
        if state_dict is not None:
            queue = self._preload_loops.get(id(state_dict), None)
            if queue is not None:
                queue.put(None)
        else:
            for queue in self._preload_loops.values():
                queue.put(None)

    def __del__(self):
        self.cancel_preload()

    def track_forward_calls(self, state_dict, total_tensor_bytes, add_prefix, verbose, preload, retain_unused, debug = False):

        out_of_order = False

        if verbose:
            import tqdm
            verbose = tqdm.tqdm(total = total_tensor_bytes, leave = False, unit = '', unit_scale = True, smoothing = 0)
            verbose.update(); verbose.close()

        if preload:
            preload_queue = queue.Queue(1)
            def preload_loop():
                if verbose:
                    nonlocal total_tensor_bytes
                    verbose.disable = False
                    verbose.reset(total_tensor_bytes)
                self._preload_loops[id(state_dict)] = preload_queue
                for tensor in range(len(state_dict)):
                    if debug:
                        print(f'thread waiting')# for {self._next_to_preload}')
                    next_tensors = preload_queue.get()
                    if next_tensors is None:
                        break
                    if debug:
                        print(f'thread got')# {self._next_to_preload}')
                    for next_tensor in next_tensors:
                        next_tensor = next_tensor.flatten()
                        for idx in range(0, len(next_tensor), mmap.PAGESIZE):
                            next_tensor[idx]
                del self._preload_loops[id(state_dict)]
                if verbose:
                    total_tensor_bytes = verbose.n
                    verbose.close()


            preload_thread = threading.Thread(target=preload_loop)
            preload_thread.start()

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

            parameter_items = [*module._parameters.items()]
            parameter_items.sort(key = lambda item: item[0])
            for tidx, (pname, parameter) in enumerate(parameter_items):
                if parameter is None:
                    continue
                stored_idx_key_tensor = stored_idx_key_tensor_by_data.get(parameter.data_ptr())
                if stored_idx_key_tensor is not None:
                    stored_idx, key, tensor = stored_idx_key_tensor

                    if verbose:
                        verbose.set_description(key)
                        verbose.update(len(tensor.flatten()) * tensor.element_size())
                    print(key, tensor.shape, module.__class__.__name__, pname)
                    # PROGRESS DATA WAS OUTPUT ABOVE LINE

                    is_last = len(module_tidx_key_tensor_by_read_idx) > 0 and tensor is module_tidx_key_tensor_by_read_idx[0][-1]

                    # set out_of_order flag
                    nonlocal out_of_order, next_stored_idx
                    if not out_of_order:
                        if stored_idx != next_stored_idx and not is_last:
                            out_of_order = True
                    next_stored_idx = (stored_idx + 1) % len(key_tensor_by_stored_idx)

                    if preload:
                        if debug:
                            print('main appending result')
                        try:
                            preload_queue.put_nowait((key_tensor_by_stored_idx[next_stored_idx][1],))
                        except queue.Full:
                            if debug:
                                print('main skipping due to last result not take')

                    # store read order
                    if not is_last:
                        module_tidx_key_tensor_by_read_idx.append((module, tidx, key, tensor))
                    else:
                        preload_queue.put(None)
                        finished_tracking_calls()
                        break

        tracking_handle = torch.nn.modules.module.register_module_forward_pre_hook(track_forward)

        def finished_tracking_calls():
            tracking_handle.remove()
            if out_of_order:
                state_dict.clear()
                state_dict.update({
                    key[len(add_prefix):] if key.startswith(add_prefix) else key : tensor
                    for module, tidx, key, tensor in module_tidx_key_tensor_by_read_idx
                })
                order_filename = self.filename + '.order.json'
                with open(order_filename, 'wt') as order_file:
                    json.dump([*state_dict.keys()], order_file)
                if retain_unused:
                    for key, tensor in key_tensor_by_stored_idx:
                        if key not in state_dict:
                            state_dict[key] = tensor
                warnings.warn(
                    f'\ntensors in {self.filename} are ordered differently from use.'
                    f'\nthey have been sorted in the read state_dict and can be resaved.'
                    f'\nruntime key order has been written to {order_filename}'
                )
            if preload:
                last_module, _, last_key, _ = module_tidx_key_tensor_by_read_idx[-1]
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
                    if debug and last_module in (prev_module, next_module):
                        # this just verifies logic is correct
                        import pdb; pdb.set_trace()
                        assert last_module not in (prev_module, next_module)
    
                    module_bin.append(prev_tensor)
                    if next_module is not prev_module:
                        # new module
                        register_module_next_tensor(read_idx - 1, last_module, last_key, module_bin)
                        module_bin.clear()
                        last_module = prev_module
                        last_key = prev_key
                module_bin.append(next_tensor)
                register_module_next_tensor(read_idx - 1, last_module, last_key, module_bin)

        def register_module_next_tensor(read_idx, module, a_key, next_tensors):
            # the tensors were smaller than ram individually; it was fine to preload them together
            bytect = sum((len(tensor.flatten()) * tensor.element_size() for tensor in next_tensors))
            label = a_key
            if '.' in label:
                label = label[:label.rfind('.')]
            def prepare_next_tensor(module, input):
                nonlocal preload_thread

                if verbose:
                    verbose.set_description(label)
                    verbose.update(bytect)

                if debug:
                     print('main2 setting result')
                try:
                    preload_queue.put_nowait(next_tensors)
                except queue.Full:
                    if debug:
                        print('main2 skipping due to last result not take')

                if not preload_thread.is_alive():
                    preload_thread = threading.Thread(target=preload_loop)
                    preload_thread.start()

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
        transformers.modeling_utils.WEIGHTS_NAME = PTMAP_WEIGHTS_NAME
        
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
        #self._transformers_pretrainedmodel_from_pretrained = transformers.modeling_utils.PreTrainedModel.__dict__['from_pretrained']
        #@classmethod
        #def from_pretrained(cls, *params, low_cpu_mem_usage = False, **kwparams):
        #    return self._transformers_pretrainedmodel_from_pretrained(cls, *params, low_cpu_mem_usage = True, **kwparams)
        #transformers.modeling_utils.PreTrainedModel.__dict__['from_pretrained'] = from_pretrained
        def Linear_wrapper(in_features, out_features, bias = True, device = None, dtype = None):
            return torch.nn.LazyLinear(out_features, bias, device, dtype)
        self._torch_nn_linear = torch.nn.Linear
        torch.nn.Linear = Linear_wrapper

    def __exit__(self, *params, **kwparams):
        torch.nn.Linear = self._torch_nn_linear
        #transformers.modeling_utils.PreTrainedModel.__dict__['from_pretrained'] = self._transformers_pretrainedmodel_from_pretrained
        transformers.pipeline= self._transformers_pipeline
        torch.save = self._torch_save
        torch.load = self._torch_load
        transformers.modeling_utils.WEIGHTS_NAME = WEIGHTS_NAME
        transformers.file_utils.WEIGHTS_NAME = WEIGHTS_NAME
        transformers.file_utils._is_offline_mode = self._transformers_offline


if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='convert a .pt file or model to a .ptmap file')
    parser.add_argument('input_path', help='.pt file or model dir to convert')
    parser.add_argument('-o', '--output_filename', required=False, help='.ptmap file to output to')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite existing files')
    parser.add_argument('-q', '--quiet', '-s', '--silent', action='store_true', help='disable saving progress meter')
    parser.add_argument('-t', '--dtype', type=str, help='cast to the provided datatype before writing')
    parser.add_argument('-m', '--map', action='store_true', help='input is a ptmap file for reoutput. page padding is removed.')
    parser.add_argument('-p', '--pagesize', type=int, help='size to align output tensors to, 1 for no alignment')
    parser.add_argument('--get-order', action='store_true', help='output is a .order.json file for copying order')
    args = parser.parse_args()

    if args.input_path.endswith('.ptmap'):
        args.map = True
    if args.output_filename.endswith('.order.json'):
        args.get_order = True
    if not args.pagesize:
        args.pagesize = PyTorchMap.pagesize

    if not args.map:
        if os.path.isdir(args.input_path):
            args.input_path = os.path.join(args.input_path, WEIGHTS_NAME)

        if not args.output_filename:
            basename = args.input_path
            if basename.endswith('.pt'):
                basename = basename[:-len('.pt')]
            elif basename.endswith('.bin'):
                basename = basename[:-len('.bin')]
            args.output_filename = basename + '.ptmap'

        outputmap = PyTorchMap(args.output_filename)

        assert not outputmap.exists() or args.force

        if not args.quiet:
            print(f'Loading {args.input_path} ...', file=sys.stderr)
        tensor_data = torch.load(args.input_path)

    elif args.map:
        if os.path.isdir(args.input_path):
            args.input_path = os.path.join(args.input_path, PTMAP_WEIGHTS_NAME)

        assert args.output_filename #or args.force

        ## this unfortunately would clobber its own memory
        #if not args.output_filename:
        #    args.output_filename = args.input_path

        inputmap = PyTorchMap(args.input_path)
        outputmap = PyTorchMap(args.output_filename)
        
        assert inputmap.exists()

        assert not outputmap.exists() or args.force

        if not args.quiet:
            print(f'Loading {args.input_path} ...', file=sys.stderr)

        tensor_data = inputmap.read(verbose=not args.quiet, writeable = True, track_forward_calls = False)

    if not args.quiet:
        print(f'Writing {args.output_filename} ...', file=sys.stderr)

    if not args.get_order:
        outputmap.write(tensor_data, pagesize=args.pagesize, verbose=not args.quiet, dtype=args.dtype)
    else:
        with open(outputmap.filename, 'wt') as outputfile:
            json.dump([*tensor_data.keys()], outputfile)
