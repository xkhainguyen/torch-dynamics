import torch
from torch.func import vmap
import numpy as np
import ipdb


# a = torch.randn((2,10))
# b = torch.randn((2,20))

# def np_convolve(x, y):
#     x, y = x.numpy(), y.numpy()
#     # Here could be some code only available on numpy array ...
#     z = np.convolve(x, y)
#     return torch.from_numpy(z)

# # @torch.compile
# def vmap_convolve(x, y):
#     return vmap(np_convolve)(x,y)

# def poormans_vmap(x, y):
#     x, y = x.numpy(), y.numpy()
#     z = np.stack([np.convolve(x1, y1) for x1, y1 in zip(x, y)])
#     return torch.from_numpy(z)

# torch.testing.assert_close(vmap_convolve(a, b), poormans_vmap(a, b))


def to_numpy(tensor):
    return tensor.cpu().numpy()

class NumpySort(torch.autograd.Function):
    @staticmethod
    def forward(x, dim):
        device = x.device
        x = to_numpy(x)
        ind = np.argsort(x, axis=dim)
        ind_inv = np.argsort(ind, axis=dim)
        result = np.take_along_axis(x, ind, axis=dim)
        return (
            torch.tensor(result, device=device),
            torch.tensor(ind, device=device),
            torch.tensor(ind_inv, device=device),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dim = inputs
        _, ind, ind_inv = output
        ctx.mark_non_differentiable(ind, ind_inv)
        ctx.save_for_backward(ind, ind_inv)
        ctx.dim = dim

    # The signature of the vmap staticmethod is:
    # vmap(info, in_dims: Tuple[Optional[int]], *args)
    # where *args is the same as the arguments to `forward`.
    @staticmethod
    def vmap(info, in_dims, x, dim):
        # For every input (x and dim), in_dims stores an Optional[int]
        # that is:
        # - None if the input is not being vmapped over or if the input
        #   is not a Tensor
        # - an integer if the input is being vmapped over that represents
        #   the index of the dimension being vmapped over.
        x_bdim, _ = in_dims

        # A "vmap rule" is the logic of how to perform the operation given
        # inputs with one additional dimension. In NumpySort, x has an
        # additional dimension (x_bdim). The vmap rule is simply
        # to call NumpySort again but pass it a different `dim`.
        x = x.movedim(x_bdim, 0)
        ipdb.set_trace()

        # Handle negative dims correctly
        dim = dim if dim >= 0 else dim + x.dim() - 1
        result = NumpySort.apply(x, dim + 1)

        # The vmap rule must return a tuple of two things
        # 1. the output. Should be the same amount of things
        #    as returned by the forward().
        # 2. one Optional[int] for each output specifying if each output
        # is being vmapped over, and if so, the index of the
        # dimension being vmapped over.
        #
        # NumpySort.forward returns a Tuple of 3 Tensors. Since we moved the
        # dimension being vmapped over to the front of `x`, that appears at
        # dimension 0 of all outputs.
        # The return is (output, out_dims) -- output is a tuple of 3 Tensors
        # and out_dims is a Tuple of 3 Optional[int]
        return result, (0, 0, 0)

def numpy_sort(x, dim=-1):
    result, _, _ = NumpySort.apply(x, dim)
    return result

x = torch.randn(2, 3)
print(x)
# result = torch.vmap(numpy_sort)(x)
# assert torch.allclose(result, numpy_sort(result, 1))

dim = 1
x = to_numpy(x)
ind = np.argsort(x, axis=dim)
ind_inv = np.argsort(ind, axis=dim)
result = np.take_along_axis(x, ind, axis=dim)

print(result)