import torch

num_classes = []
c = torch.zeros((20, 3))
c[:10, -1] = 1
c[10:, 1] = 1

a = torch.tensor([[0, 1, 0], [1, 1, 0]])
result = torch.vstack((c, a))

sums = torch.sum(result, dim=0)
weights = 1 / sums
weights = weights / weights.min()
examples_weights = result * weights
print(examples_weights.max(dim=-1))

# mean_weights = torch.zeros(examples_weights.shape[0])
# for i, example_weight in enumerate(examples_weights):
#     mask = example_weight != 0
#     mean_weights[i] = example_weight[mask].max()
# print(mean_weights)
# print(examples_weights[examples_weights != 0, :].mean(dim=0))
# print((result * weights).sum(dim=-1))
# samples_weight = torch.tensor([weight[t] for t in targets])
# return samples_weight
