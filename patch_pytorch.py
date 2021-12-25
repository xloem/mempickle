import torch
import numpy as np
import os

if os.uname().machine == 'aarch64' and not torch.cuda.is_available():
    # some torch functions crash on low-end systems
    # ( torch 1.10.1 https://github.com/pytorch/pytorch/issues/70350 )
    # this module attempts to patch them to use numpy functions instead

    def patch(*modules):
        def patch(func):
            for module in modules:
                setattr(module, func.__name__, func)
            return func
        return patch
    
    @patch(torch)
    def mm(mat1, mat2, *, out=None):
        if out is None:
            out = torch.einsum('ij,jk->ik', mat1, mat2)
        else:
            out[:] = torch.einsum('ij,jk->ik', mat1, mat2)
        return out
    
    @patch(torch)
    def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        out = mm(mat1, mat2, out=out)
        out *= alpha
        if beta != 0:
            out += input * beta
        return out
    
    @patch(torch._C._nn)
    def linear(input, weight, bias = None):
        weight_rows = weight.shape[0]
        out = mm(input.view((-1, input.shape[-1])), weight.T)
        out = out.view((*input.shape[:-1], weight_rows))
        if bias is not None:
            out += bias
        return out
