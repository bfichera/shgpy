import numpy as np
import sympy as sp

def union(*array):
    ans = []
    for y in array:
        for x in y:
            if x not in ans:
                ans.append(x)
    return ans

def tensor_product(*tensors):
    if len(tensors) == 2:
        return np.tensordot(tensors[0], tensors[1], axes=0)
    else:
        return np.tensordot(tensors[0], tensor_product(*tensors[1:]), axes=0)

def tensor_contract(tensor, index_pairs):
    ans = tensor
    index_idx = list(range(len(np.shape(tensor))))
    index_pairs = index_pairs[:]
    while len(index_pairs) > 0:
        old_index_pair = index_pairs.pop(0)
        new_index_pair = [index_idx.index(old_index_pair[i]) for i in range(len(old_index_pair))]
        ans = np.trace(ans, axis1 = new_index_pair[0], axis2 = new_index_pair[1])
        for i in range(len(old_index_pair)):
            index_idx.remove(old_index_pair[i])
    return ans
    
def transform(tensor, operation):
    rank = len(tensor.shape)
    args = [operation]*rank
    args.append(tensor)
    return tensor_contract(tensor_product(*args), [[2*i+1, 2*rank+i] for i in range(rank)])
