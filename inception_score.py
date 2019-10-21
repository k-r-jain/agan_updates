import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import inception_v3


net = inception_v3(pretrained=True).cuda()


def inception_score(images, batch_size=5):
    scores = []
    for i in range(int(math.ceil(float(len(images)) / float(batch_size)))):
        batch = Variable(torch.cat(images[i * batch_size: (i + 1) * batch_size], 0))
    # for batch in ds:
        # print(batch.size())
        s, _ = net(batch)  # skipping aux logits
        scores.append(s)
    p_yx = F.softmax(torch.cat(scores, 0), 1)
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
    final_score = KL_d.mean()
    return final_score


from torchvision import datasets
import torchvision.transforms as transforms

cifar = datasets.CIFAR10(root = '/home/kartik/data', download = True, transform=transforms.Compose([transforms.Scale(32), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))
cifar_loader = torch.utils.data.DataLoader(cifar, batch_size = 64)
print(inception_score(cifar.size(), batch_size = 64))


# if __name__ == '__main__':
#     class IgnoreLabelDataset(torch.utils.data.Dataset):
#         def __init__(self, orig):
#             self.orig = orig

#         def __getitem__(self, index):
#             return self.orig[index][0]

#         def __len__(self):
#             return len(self.orig)

#     import torchvision.datasets as dset
#     import torchvision.transforms as transforms

#     cifar = dset.CIFAR10(root='data/', download=True,
#                              transform=transforms.Compose([
#                                  transforms.Scale(32),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                              ])
#     )

#     IgnoreLabelDataset(cifar)

#     print ("Calculating Inception Score...")
#     print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=64, resize=True, splits=10))