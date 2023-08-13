import torch

dict = torch.load(".pth", map_location='cuda:0')
old = []
for d in dict['model_state_dict']:
    old.append(d)
new = []
for i in old:
    new.append(i[7:])
key_name_mapping = {old: new for old, new in zip(old, new)}


def batch_rename_keys(original_dict, key_mapping):
    """
    批量更改字典的键名
    参数:
        original_dict (dict): 原始的字典，包含需要更改键名的键值对。
        key_mapping (dict): 键名映射字典，包含旧键名到新键名的映射关系。
    返回:
        dict: 返回包含更新后键名的新字典。
    """
    updated_dict = {'model_state_dict':{}, 'optimizer_state_dict':{}}

    for key, value in original_dict['model_state_dict'].items():
        if key in key_mapping:
            new_key = key_mapping[key]
        else:
            new_key = key
        updated_dict['model_state_dict'][new_key] = value
        
    for key, value in original_dict['optimizer_state_dict'].items():
        updated_dict['optimizer_state_dict'][key] = value

    return updated_dict


updated_dict = batch_rename_keys(dict, key_name_mapping)
torch.save(updated_dict, '.pth')

