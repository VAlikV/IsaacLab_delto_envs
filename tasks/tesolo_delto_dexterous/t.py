import torch

# a = torch.zeros((10,5,3))

b = torch.randint(-15, 15, (10, 5))

a = b
a = a * 2

# u = torch.tensor([1, 2, 3, 4, 5])
# l = torch.tensor([-1, -2, -3, -4, -5])

# c = torch.clip(b,l, u)

# d = torch.cat([u,l])

# a[2,0:3] = 2
# a[4,3:4] = 5
# a[7,2:5] = 3
# a[9,0:5] = 5

# c = (a - b.unsqueeze(1)).norm(dim=2).sum(dim=1)

# b=torch.unsqueeze(torch.max(a, dim=1)[0], dim=1)
# b = torch.where(b > 2, b, 0)
# c = torch.norm(a, dim=1)
print(a)
print(b)
# print(c)
# print(d)