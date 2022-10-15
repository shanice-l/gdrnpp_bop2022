import torch
import os.path as osp
import sys
from torch.autograd import Variable

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)
import torch_nndistance as NND


p1 = torch.rand(10, 1000, 3)
p2 = torch.rand(10, 1500, 3)
points1 = Variable(p1, requires_grad=True)
points2 = p2
points1 = points1.cuda()
print(points1.requires_grad)
points2 = points2.cuda()
dist1, dist2 = NND.nnd(points1, points2)
print(dist1, dist2)
loss = torch.sum(dist1)
print("loss", loss)
loss.backward()
print(points1.grad, points2.grad)

print("====================")
points1 = Variable(p1.cuda(), requires_grad=True)
points2 = p2.cuda()
dist1, dist2 = NND.nnd(points1, points2)
print(dist1, dist2)
loss = torch.sum(dist1)
print("loss", loss)
loss.backward()
print(points1.grad, points2.grad)
