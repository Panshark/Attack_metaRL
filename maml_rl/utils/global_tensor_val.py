import torch

def _init():
    global _global_dict
    _global_dict = {}

def set_tensor_value(name, value):
    _global_dict[name] = torch.tensor(value, requires_grad=True)

def get_value(name, defValue=None):
    Get_value = _global_dict
    try:
        return Get_value[name]
    except KeyError:
        return defValue