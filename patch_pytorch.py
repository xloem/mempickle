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
        return patch
    
    @patch(torch)
    def mm(mat1, mat2, *, out=None):
        if out is None:
            out = torch.empty((mat1.shape[0], mat2.shape[1]))
        np.matmul(mat1.numpy(), mat2.numpy(), out=out.numpy())
        return out
    
    @patch(torch)
    def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        if out is None:
            out = torch.empty((mat1.shape[0], mat2.shape[1]))
        np.matmul(mat1.numpy(), mat2.numpy(), out=out.numpy())
        out *= alpha
        if beta != 0:
            out += input * beta
        return out
    
    @patch(torch._C._nn)
    def linear(input, weight, bias = None):
        weight_rows = weight.shape[0]
        out = torch.empty((*input.shape[:-1], weight_rows))
        np.matmul(
            input.view((-1, input.shape[-1])).numpy(),
            weight.T.numpy(),
            out=out.view((-1, weight_rows)).numpy()
        )
        if bias is not None:
            out += bias
        return out
