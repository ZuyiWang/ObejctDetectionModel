import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset

from torch.utils.tensorboard import SummaryWriter
import numpy as np

use_gpu = torch.cuda.is_available()

file_root = '/home/wangzy/ObjectDetectionModel/Data/VOCdevkit/VOC2012_tra_val/JPEGImages'
test_file_root = '/home/wangzy/ObjectDetectionModel/Data/VOCdevkit/VOC2007_test/JPEGImages'
log_dir = './runs'
learning_rate = 0.001
num_epochs = 50
batch_size = 24
use_resnet = True
if use_resnet:
    net = resnet50(pretrained=True, model_pth='./resnet50-19c8e357.pth')
else:
    net = vgg16_bn()
# net.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             #nn.Linear(4096, 4096),
#             #nn.ReLU(True),
#             #nn.Dropout(),
#             nn.Linear(4096, 1470),
#         )
#net = resnet18(pretrained=True)
#net.fc = nn.Linear(512,1470)
# initial Linear
# for m in net.modules():
#     if isinstance(m, nn.Linear):
#         m.weight.data.normal_(0, 0.01)
#         m.bias.data.zero_()
#net.load_state_dict(torch.load('yolo.pth'))
# print(net)

# if use_resnet:
#     resnet = models.resnet50(pretrained=True)
#     new_state_dict = resnet.state_dict()
#     dd = net.state_dict()
#     for k in new_state_dict.keys():
#         print(k)
#         if k in dd.keys() and not k.startswith('fc'):
#             print('yes')
#             dd[k] = new_state_dict[k]
#     net.load_state_dict(dd)
# else:
    # vgg = models.vgg16_bn(pretrained=True)
    # new_state_dict = vgg.state_dict()
    # dd = net.state_dict()
    # for k in new_state_dict.keys():
    #     print(k)
    #     if k in dd.keys() and k.startswith('features'):
    #         print('yes')
    #         dd[k] = new_state_dict[k]
    # net.load_state_dict(dd)

if False:
    net.load_state_dict(torch.load('best.pth'))
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(7,2,5,0.5)
if use_gpu:
    net.cuda()  # set environment 'CUDA_VISIBLE_DEVICES'

net.train() # change mode
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

# train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
train_dataset = yoloDataset(root=file_root,list_file='voc2012_tra_val.txt',train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

# test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
test_dataset = yoloDataset(root=test_file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=4)

print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
best_test_loss = np.inf
writer = SummaryWriter(log_dir)
add_graph = False


for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate=0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    if epoch == 40:
        learning_rate=0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)

    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    writer.add_scalar('Learning Rate Now', learning_rate, epoch+1)

    
    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        # Add graph
        if not add_graph:
            writer.add_graph(net, images)
            add_graph = True

        pred = net(images)
        loss = criterion(pred,target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1
            writer.add_scalar('average_loss', total_loss / (i+1), epoch * len(train_loader) + i)
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

    #validation
    with torch.no_grad():
        validation_loss = 0.0
        net.eval()
        for i,(images,target) in enumerate(test_loader):
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            
            pred = net(images)
            loss = criterion(pred,target)
            validation_loss += loss.item()
        validation_loss /= len(test_loader)
        writer.add_scalar('val loss', validation_loss, epoch)
        
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(),'best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
        logfile.flush()      
torch.save(net.state_dict(),'yolo.pth')