import pytorch_tensormap
import torch, transformers

def load_model(model, writeable = False):
    if type(model) is str:
        config = transformers.models.auto.configuration_auto.AutoConfig.from_pretrained(model)
    else:
        config = model.config
    with pytorch_tensormap.Ctx(writeable = writeable):
        framework, model = transformers.pipelines.infer_framework_load_model(model, config, low_cpu_mem_usage = True)
    return framework, model

class stype:
    def __init__(self, dtype):
        self.dtype = dtype
        self.size = torch.tensor(0, dtype=self.dtype).element_size()

def walk_module_tree(name_parts, mod, desc_visitor = lambda name_parts, mod: None, asc_visitor = lambda name_parts, mod: None):
    desc_visitor(name_parts, mod)
    for subname, submod in mod.named_children():
        walk_module_tree((*name_parts, subname), submod, desc_visitor, asc_visitor)
    asc_visitor(name_parts, mod)

class Retrainer:
    def __init__(self, src_model, dst_model, mem_limit = 512*1024*1024):
        self.mem_limit = mem_limit
        self.models = {}
        _, self.models['src'] = load_model(src_model)
        self.models['src'].eval()
        _, self.models['dst'] = load_model(dst_model, writeable = True)

        self.modules_by_shallowness = {}
        
        for name, mod in self.models.items():
            max_depth = 0
            def desc_visitor(name_parts, mod):
                nonlocal max_depth
                max_depth = max(max_depth, len(name_parts))
            walk_module_tree((name,), mod, desc_visitor=desc_visitor)
            self.modules_by_shallowness[name] = [{} for x in range(max_depth+1)]
            def desc_visitor(name_parts, mod):
                shallowness = max_depth - len(name_parts)
                if type(mod) is not torch.nn.Dropout:
                    subname = '.'.join(name_parts[1:])
                    self.modules_by_shallowness[name][shallowness][subname] = mod
            walk_module_tree((name,), mod, desc_visitor=desc_visitor)
    def datagen_for_mod(self, mod):
        input_const_factor = 0
        calc_const_factor = 0
        output_const_factor = 0
        input_value_scale = 1
        integer = False
        data_shape_factors = 2
        if type(mod) is torch.nn.Embedding:
            input_stype = stype(torch.long)
            input_value_scale = mod.num_embeddings
            integer = True
            calc_stype = stype(mod.weight.dtype)
            input_shape = ()
            calc_scale_factor = len(mod.weight.flatten()) 
            output_shape = (mod.embedding_dim,)
        elif type(mod) is torch.nn.LayerNorm:
            input_stype = stype(mod.bias.dtype)
            input_shape = mod.normalized_shape
            calc_stype = input_stype
            calc_scale_factor = len(mod.weight.flatten()) * 3 # guess
            output_shape = input_shape
        elif type(mod) is transformers.Conv1D:
        #else:
            input_stype = stype(mod.weight.dtype)
            input_shape = (mod.weight.shape[0],)
            calc_stype = input_stype
            calc_scale_factor = mod.weight.shape[0] * mod.weight.shape[1] + mod.bias.shape[0]
            output_shape = mod.bias.shape
        elif hasattr(mod, 'embed_dim'):
            input_shape = (mod.embed_dim,)
            output_shape = input_shape
            input_stype = None
            calc_scale_factor = mod.embed_dim * mod.embed_dim # guess
            for parameter in mod.parameters():
                if input_stype is None:
                    input_stype = stype(parameter.dtype)
                    break
            calc_stype = input_stype
        elif hasattr(mod, 'ln_1'):
            input_shape = (*mod.ln_1.bias.shape,)
            output_shape = input_shape
            input_stype = None
            calc_scale_factor = 1
            for dim in input_shape:
                calc_scale_factor *= dim * dim # guess
            for parameter in mod.parameters():
                if input_stype is None:
                    input_stype = stype(parameter.dtype)
                    break
            calc_stype = input_stype
        #else:
        #    raise AssertionError(f'unimplemented module type {type(mod)}')
        else:
            import pdb; pdb.set_trace()
            import warnings; warnings.warn(f'unimplemented module type {type(mod)}')
            input_stype = None
            calc_scale_factor = 0
            if hasattr(mod, 'bias'):
                input_stype = stype(mod.bias.dtype)
                input_shape = mod.bias.shape
            for parameter in mod.parameters():
                if input_stype is None:
                    input_stype = stype(parameter.dtype)
                    input_shape = parameter.shape
                calc_scale_factor += len(parameter.flatten())
            calc_stype = input_stype
            output_shape = input_shape

        input_scale_factor = input_stype.size
        for input_dim in input_shape:
            input_scale_factor *= input_dim
            
        calc_const_factor *= calc_stype.size
        calc_scale_factor *= calc_stype.size

        output_scale_factor = calc_stype.size
        for output_dim in output_shape:
            output_scale_factor *= output_dim


        output_const_factor *= 4
        output_scale_factor *= 4

        extra_count = (self.mem_limit - input_const_factor - calc_const_factor - output_const_factor) / (input_scale_factor + calc_scale_factor + output_scale_factor)

        assert extra_count > 4

        extra_count = int(extra_count ** 0.5)

        input_shape = (extra_count, extra_count + 1, *input_shape)

        if integer:
            def rand():
                return torch.randint(input_value_scale, input_shape, dtype=input_stype.dtype)
        else:
            def rand():
                return torch.randn(input_shape, dtype=input_stype.dtype) * input_value_scale
        return rand
    #def compare_mod(self, modname):
    #    return self.compare_mods(modname, modname)
    def compare_mods(self, src_module, dst_module):
        #if type(src_module) is str:
        #    src_module = self.src_modules[src_module]
        #if type(dst_module) is str:
        #    dst_module = self.dst_modules[dst_module]

        datagen = self.datagen_for_mod(src_module)
        data = datagen()
        with torch.no_grad():
            out1 = src_module(data)
        out2 = dst_module(data)
        if type(out1) is tuple:
            out1 = torch.stack([elem for elem in out1 if elem is not None])
        if type(out2) is tuple:
            out2 = torch.stack([elem for elem in out2 if elem is not None])
        return out1, out2

    def mods_by_difference(self, shallowness = 0):
        self.models['dst'].eval()
        src_modules = self.modules_by_shallowness['src'][shallowness]
        dst_modules = self.modules_by_shallowness['dst'][shallowness]
        mods = []
        with torch.no_grad():
            for (src_name, src_module), (dst_name, dst_module) in zip(src_modules.items(), dst_modules.items()):
                assert src_name == dst_name
                out1, out2 = self.compare_mods(src_module, dst_module)
                diff = (out1 - out2).abs().mean()
                #print(src_name, dst_name, diff)
                mods.append((diff, (src_name, src_module), (dst_name, dst_module)))
        mods.sort(key=lambda mod_entry: mod_entry[0], reverse=True)
        return mods

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='train one model to produce the same output as another')
    parser.add_argument('input_dir', help='input model dir')
    parser.add_argument('output_dir', help='output model dir')
    args = parser.parse_args()

    retrainer = Retrained(args.input_dir, args.output_dir)

