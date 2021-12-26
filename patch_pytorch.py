import torch
import os
import sys

def mm_aarch64_succeeds():
    pid = os.fork()

    if pid == 0:
        # in child process
        try:
            # crashing operation
            torch.mm(torch.randn((4096,4096)), torch.randn((4096,4096)))
        except:
            sys.exit(-1)
        else:
            sys.exit(0)

    elif pid != 0:
        # in parent process
        pid, status = os.waitpid(pid, 0)
        status = os.waitstatus_to_exitcode(status)
        return status == 0

if (
    os.uname().machine == 'aarch64'
    and not torch.cuda.is_available()
    and not mm_aarch64_succeeds()
):
    import warnings

    warnings.warn(
        '\ntorch matmul is crashing on your arch, likely due to blas compilation flags'
        '\nsee e.g. https://github.com/pytorch/pytorch/issues/70350'
        '\nsome functions will be patched to perform matmul via numpy'
        '\nif you get an illegal instruction crash, some function was not patched'
        '\nthis workaround uses ctypes and bfloat16 for half-precision math'
    )

    import bfloat16, ctypes, numpy as np

    def patch(*modules, aliases = ()):
        if type(aliases) is str:
            aliases = aliases.split(' ')
        def patch(func):
            for module in modules:
                setattr(module, func.__name__, func)
                for alias in aliases:
                    setattr(module, alias, func)
            return func
        return patch

    def npbfloat16(t):
        bytect = len(t.flatten()) * 2
        carray = (ctypes.c_char * bytect).from_address(t.data_ptr())
        return np.ndarray(shape=t.shape, buffer=carray, dtype=bfloat16.bfloat16)
    
    @patch(torch, aliases='matmul')
    def mm(mat1, mat2, *, out=None):
        ## numpy matmul does not crash.  bfloat16 needs an additional package. ##
        if out is None:
            out = torch.empty((*mat1.shape[:-2], mat1.shape[-2], mat2.shape[-1]), dtype = mat1.dtype)
        if mat1.dtype is torch.bfloat16:
            mat1, mat2, npout = npbfloat16(mat1), npbfloat16(mat2), npbfloat16(out)
            npout[:] = np.matmul(mat1, mat2)
        else:
            mat1, mat2, npout = mat1.numpy(), mat2.numpy(), out.numpy()
            np.matmul(mat1, mat2, out=npout)
        return out
        ## mm crashes ##
        ## einsum crashes ##
        #if out is None:
        #    out = torch.einsum('ij,jk->ik', mat1, mat2)
        #else:
        #    out[:] = torch.einsum('ij,jk->ik', mat1, mat2)
        #return out
        ## tensordot crashes ##
        #if out is None:
        #    out = torch.empty((mat1.shape[0], mat2.shape[1]))
        #return torch.tensordot(mat1, mat2, dims=1, out=out)
        ## inner crashes ##
        #return torch.inner(mat1, mat2.T, out=out)
        ## mv crashes ##
        #if out is None:
        #    out = torch.empty((mat1.shape[0], mat2.shape[1]))
        #for colidx in range(out.shape[1]):
        #    torch.mv(mat1, mat2[:, colidx], out = out[:, colidx])
        #return out
        ## 1d dot product does not crash as much but can still crash! ##
        ## no crash experienced with bfloat16 but it was pretty slow ##
        #if out is None:
        #    out = torch.empty((mat1.shape[0], mat2.shape[1]), dtype = mat1.dtype)
        #for mat1row, outrow in zip(mat1, out):
        #    for mat2col, outpos in zip(mat2.T, outrow):
        #        torch.dot(mat1row, mat2col, out=outpos)
        #return out
        ## broadcast multiplication does not crash, but can get large and slow ##
        #if out is None:
        #    out = (
        #        # construct huge thing,
        #        mat1.view(mat1.shape[0], 1, mat1.shape[1]) *
        #        mat2.T.broadcast_to(mat1.shape[0], *mat2.T.shape)
        #    ).sum(axis=2) # construct smaller thing
        #else:
        #    out[:] = (
        #        # construct huge thing,
        #        mat1.view(mat1.shape[0], 1, mat1.shape[1]) *
        #        mat2.T.broadcast_to(mat1.shape[0], *mat2.T.shape)
        #    ).sum(axis=2) # construct smaller thing
        #return out
    
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
