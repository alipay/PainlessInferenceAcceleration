
import torch


# model_name = '/mntnlp/common_base_model/opt_6b7/pytorch_model.bin'
# model_name = '/mnt_alipayshnas/workspace/nanxiao/opt_kl_sft_7/pytorch_model_old_mlp_name.bin'
# sd = torch.load(model_name)


# state_dict = {}
# for k,v in sd.items():
#     if 'fc1' in k:
#         k = k.replace('.fc1','.mlp.fc1')
#     if 'fc2' in k:
#         k = k.replace('.fc2','.mlp.fc2')
#     state_dict[k] = v
# print(state_dict.keys())
# torch.save(state_dict, '/mnt_alipayshnas/workspace/nanxiao/opt_kl_sft_7/pytorch_model.bin')


sd = torch.load('/mnt_alipayshnas/workspace/nanxiao/opt_kl_mlp_7/pytorch_model.bin')
print(sd.keys())

