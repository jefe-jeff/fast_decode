data = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13]])
data = data.unfold(1, 3, 1)

weight1 = torch.rand((4, 1, 3, 5))
data = data.unsqueeze(3)
layer = data * weight1
layer = layer.sum(2)
layer = layer.clamp(0)

weight2 = torch.rand((4, 1, 5, 10))
layer = layer.unsqueeze(3)
layer = layer * weight2
layer = layer.sum(2)
layer = layer.clamp(0)

weight3 = torch.rand((4, 1, 10, 1))
layer = layer.unsqueeze(3)
layer = layer * weight3
layer = layer.sum(2)

output= layer.squeeze()