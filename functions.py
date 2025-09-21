import torch

def fixed_digits(x, n):
    return f"{x:.{n}f}"

def labels_to_tensor(y):
    lst = y.tolist()
    if type(lst) != list:
        lst = [lst]
    length = len(lst)
    tensor = torch.zeros(length, 10)
    row = 0
    for i in lst:
        tensor[row, int(i)] = 1
        row += 1
    return tensor