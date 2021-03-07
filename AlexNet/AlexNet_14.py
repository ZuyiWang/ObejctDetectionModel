import torch
import torchvision
import torchvision.models as models

imagenet_data = torchvision.datasets.ImageNet('/home/wangzy/ObjectDetectionModel/ImageNet')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True)

alexnet = models.alexnet()


