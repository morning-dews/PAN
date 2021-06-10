import argparse
import os
import numpy as np
import math

# import torchvision.transforms as transforms
# from torchvision.utils import save_image

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable
# import helper

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import pdb

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--dropoutrate', type=float, default=0.3, help='dropoutrate')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--gpu', type=int, default=0, help='gpu no.')
parser.add_argument('--model', type=str, default='pan', choices=['pan', 'nnpu','upu', 'agan'], help='Please select model that you want to run.')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10','sddd', 'news'], help='Using dataset.')
opt = parser.parse_args()
print(opt)
model_style = {'mnist':'mlp','cifar10':'conv', 'sddd':'mlp', 'news':'mlp' }
img_shape = (opt.channels, opt.img_size, opt.img_size)
img_shape = (1 , 28, 28)
torch.cuda.set_device(opt.gpu)
cuda = True if torch.cuda.is_available() else False

def adjust_lr(optimizer, epoch):
    lr = opt.lr
    lr1 = opt.lr * (0.5 ** (epoch // 20))
    if lr == lr1:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr1


class Discriminator_MLP(nn.Module):
    def __init__(self):
        super(Discriminator_MLP, self).__init__()
        # dense1_bn = nn.BatchNorm1d(512)
        # dense2_bn = nn.BatchNorm1d(256)


        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropoutrate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropoutrate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class Recognizer_MLP(nn.Module):
    def __init__(self):
        super(Recognizer_MLP, self).__init__()
        # dense1_bn = nn.BatchNorm1d(512)
        # dense2_bn = nn.BatchNorm1d(256)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropoutrate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropoutrate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class Recognizer_CNN(nn.Module):
    def __init__(self):
        super(Recognizer_CNN, self).__init__()
        # dense1_bn = nn.BatchNorm1d(512)
        # dense2_bn = nn.BatchNorm1d(256)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.BatchNorm2d(96),
            nn.Dropout(opt.dropoutrate),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(96, 96, 3, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.BatchNorm2d(96),
            nn.Dropout(opt.dropoutrate),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(96, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            # nn.BatchNorm2d(192),
            nn.Dropout(opt.dropoutrate),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, 10, 1),
            nn.ReLU(),
            # nn.BatchNorm2d(10),
            nn.Dropout(opt.dropoutrate),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(196*10, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(opt.dropoutrate),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        conv_d = self.conv(img)
        out = self.fc1(conv_d.view(conv_d.shape[0],-1))

        return out


class Discriminator_CNN(nn.Module):
    def __init__(self):
        super(Discriminator_CNN, self).__init__()
        # dense1_bn = nn.BatchNorm1d(512)
        # dense2_bn = nn.BatchNorm1d(256)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 3),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(opt.dropoutrate),
            nn.Conv2d(96, 96, 3, stride=2),
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout(opt.dropoutrate),
            nn.Conv2d(96, 192, 1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Dropout(opt.dropoutrate),
            # nn.BatchNorm2d(96),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, 10, 1),
            nn.ReLU(),
            # nn.BatchNorm2d(10),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(196*10, 1000),
            # nn.ReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(opt.dropoutrate),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        conv_d = self.conv(img)
        out = self.fc1(conv_d.view(conv_d.shape[0],-1))

        return out

def index2label(index):
    return 2*index.float() - 1


def rl(score_r):
    ###score_r : 300*1
    p_pos, p_neg = score_r, 1-score_r
    p = torch.cat((p_neg, p_pos),1)
    #### p : 300*2
    
    max_prob, max_index = torch.max(p,1)
    ####max_prob : 300*1
    ####max_index : 300*1 (0/1)
    normal_index = max_index.float().unsqueeze(1)
    reverse_index = 1.0 - normal_index
    action_index = torch.cat((reverse_index, normal_index),1) 
    
    neg_log_prob =  torch.log(max_prob)

    ###change to nll
    label = index2label(max_index)
    ####label: 300*1 (-1/1)
    #print('mmm',max_prob)
    #print('label',label) 
    max_prob = max_prob * label
    #print('max_prob',max_prob.size())   
    return  max_prob.unsqueeze(1), action_index


cfg_path = './cfg/{}'.format(opt.dataset)
pn_cfg = helper.read_cfg_file(os.path.join(cfg_path, 'PN'))
upu_cfg = helper.read_cfg_file(os.path.join(cfg_path, 'uPU'))
nnpu_cfg = helper.read_cfg_file(os.path.join(cfg_path, 'nnPU'))
assert \
    upu_cfg['dataset']['dataset_name'] == \
    nnpu_cfg['dataset']['dataset_name'] and \
    upu_cfg['network']['network_name'] == \
    nnpu_cfg['network']['network_name'] and \
    pn_cfg['dataset']['dataset_name'] == \
    nnpu_cfg['dataset']['dataset_name'] and \
    pn_cfg['network']['network_name'] == \
    nnpu_cfg['network']['network_name']
exp_name = 'exp_{}_{}_{}'.format(
    nnpu_cfg['dataset']['dataset_name'],
    nnpu_cfg['network']['network_name'],
    helper.get_unique_name()
)
log_data = helper.LogData()

# upu and nnpu.
PuDataset, PnDataset = helper.load_dataset(upu_cfg)
pu_dataset = PuDataset(upu_cfg['dataset'])
training_iterator = pu_dataset.get_training_iterator()

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
# generator = Generator()
if model_style[opt.dataset] == 'mlp':
    discriminator = Discriminator_MLP()
    recognizer = Recognizer_MLP()
else:
    discriminator = Discriminator_CNN()
    recognizer=Recognizer_CNN()

if cuda:
    # generator.cuda()
    discriminator.cuda()
    recognizer.cuda()
    # adversarial_loss.cuda()

# Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=1e-3)
optimizer_R = torch.optim.Adam(recognizer.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=1e-3)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
TensorB = torch.cuda.ByteTensor if cuda else torch.ByteTensor

# ----------
#  Training
# ----------
epoch = 0
i = 0
eps = 1e-2
gamma = 2.0
beta = 10 # 10 for mnist
epoch_num = 0
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
big_acc = 100
big_f = 0


for imgs in training_iterator:

    # Adversarial ground truths
    imgs_data = Variable(Tensor(imgs[0]), requires_grad=False)
    # print(torch.max(imgs_data))
    imgs_label = Variable(Tensor(imgs[1]), requires_grad=False)
    positive_num = int(torch.sum(imgs_label).data)

    # imgs_u_data = Tensor(imgs[0][0])
    # imgs_u_label = Tensor(imgs[1][0])/
    # imgs_l_data = Tensor(imgs[0][1])
    # imgs_l_label = Tensor(imgs[1][1])

    # -----------------
    #  Train Generator
    # -----------------


    

    if opt.model == 'nnpu':
        score_D = discriminator.forward(imgs_data)
        score_R = recognizer.forward(imgs_data)

        loss_p = 0.5 * torch.sum((1 - score_R) * imgs_label) / torch.sum(imgs_label)
        loss_u = torch.sum(score_R * (1 - imgs_label)) / torch.sum(1 - imgs_label) - 0.5 * torch.sum((score_R) * imgs_label) / torch.sum(imgs_label)

        objective = loss_p + loss_u
        if loss_u.data < -0:
            objective = loss_p - 0
            out = - loss_u
        else:
            out = objective
        loss_max_d = objective
        optimizer_R.zero_grad()
        out.backward()
        optimizer_R.step()

    if opt.model == 'upu':
        score_D = discriminator.forward(imgs_data)
        score_R = recognizer.forward(imgs_data)

        loss_p = 0.5 * torch.sum((1 - score_R) * imgs_label) / torch.sum(imgs_label)
        loss_u = torch.sum(score_R * (1 - imgs_label)) / torch.sum(1 - imgs_label) - 0.5 * torch.sum((score_R) * imgs_label) / torch.sum(imgs_label)

        objective = loss_p + loss_u
        # if loss_u.data < -0:
        #     objective = loss_p - 0
        #     out = - loss_u
        # else:
        out = objective
        loss_max_d = objective
        optimizer_R.zero_grad()
        out.backward()
        optimizer_R.step()

    if opt.model == 'pan':

        unlabeled_data = imgs_data[imgs_label.view(-1)==0]

        score_R = recognizer.forward(unlabeled_data)
        score_D = discriminator.forward(imgs_data)
        # score_R = recognizer.forward(imgs_data)
        score_D_p = score_D[imgs_label.view(-1)==1]
        score_D_u = score_D[imgs_label.view(-1)==0]
        score_D_u_h = score_D_u[:score_D_p.shape[0]]
        score_R_h = score_R[:score_D_p.shape[0]]

        R_rate = score_R.clone().detach()
        d_rate_u = score_D_u_h.clone().detach()
        

        ######################################################
        ######################################################
        # please tunning the following hyper-params as different dataset may have different suitable hyper-params.
        ######################################################
        ######################################################

        # pdb.set_trace()
        #loss = imgs_label * torch.log(score_D + eps) + 0.01 * (1 - imgs_label) * (1 - R_rate) * torch.log( 1.0 - score_D + eps) + \
        #        0.0003 * w * (1 - imgs_label) * (1 - score_R + eps + 0.0) * torch.tan(1.5 * (gamma * d_rate - 1))
	#loss = imgs_label * torch.log(score_D + eps) + 0.01 * (1 - imgs_label) * (1 - R_rate) * torch.log( 1.0 - score_D + eps) + \
         #        0.0003 * w * (1 - imgs_label) * (1 - score_R + eps + 0.0) * torch.tan(1.5 * (gamma * d_rate - 1))
        #loss = imgs_label * torch.log(score_D + eps) + 0.000 * (1 - imgs_label) * torch.log(
        #    1.0 - score_D + eps) + w * 0.001 * (1 - imgs_label) * (1 * torch.log(1 - score_R + eps + 0.0) - torch.log(score_R + eps + 0.0)) * (score_D - torch.mean(d_rate))
        
        loss1 = torch.sum(torch.log(score_D_p + eps)) + 1.0 * torch.sum( torch.max(torch.log(1.0 - score_D_u_h + eps)-torch.log(1.0 - torch.mean(d_rate_u)), torch.zeros(d_rate_u.shape).cuda()) )

        loss2 =  0.0001 * torch.sum((1 * torch.log(1 - score_R + eps + 0.0) - torch.log(score_R + eps + 0.0)) * (2 * score_D_u - 1.0 )) #torch.mean(d_rate_u)
        
        loss = loss1 + loss2

        #loss = imgs_label * torch.log(score_D + eps) + 0.02 * (1 - imgs_label) * torch.max(torch.log(1.0 - score_D + eps)-torch.log(1.0 - torch.mean(d_rate)), torch.zeros(d_rate.shape).cuda() - 10000.0) + w     * 0.001 * (1 - imgs_label) * (score_D * torch.log(score_D + eps) + (1 - score_D)* torch.log(1 - score_D + eps) - score_D * torch.log(score_R + eps) - (1 - score_D) * torch.log(1 - score_R + eps) )
	#loss = imgs_label * torch.log(score_D + eps) + 0.001 * (1 - imgs_label) * torch.log(
         #     1.0 - score_D + eps) + w * 0.0001 * (1 - imgs_label) * (1 * torch.log(1 - score_R + eps + 0.0) - torch.log(score_R + eps + 0.0)) * (score_D - torch.mean(d_rate))

        #for cifar
        #loss = imgs_label * torch.log(score_D + eps) + 0.05 * (1 - imgs_label) * torch.max(torch.log(1.0 - score_D + eps)-torch.log(1.0 - torch.mean(d_rate)), torch.zeros(d_rate.shape).cuda()) + w * 0.0001 * (1 - imgs_label) * (1 * torch.log(1 - score_R + eps + 0.0) - torch.log(score_R + eps + 0.0)) * (2 * score_D - 1.0)

        #for sdd
        # loss = imgs_label * torch.log(score_D + eps) + 0.1 * (1 - imgs_label) * torch.log(
        #     1.0 - score_D + eps) + w * 0.005 * (1 - imgs_label) * (1 * torch.log(1 - score_R + eps + 0.0) - torch.log(score_R + eps + 0.0)) * (score_D - torch.mean(d_rate))


        # loss = imgs_label * torch.log(score_D + eps) + 0.01 * (1 - imgs_label) * torch.log(
        #     1.0 - score_D + eps) + w * 0.0001 * (1 - imgs_label) * (
        #                    torch.log(1 - score_R + eps + 0.0) - 1.0 * torch.log(score_R + eps + 0.0)) * (
        #         score_D - torch.mean(d_rate))
        #
        # loss_1 =  imgs_label * (1.0 - 0.01 * score_R) * torch.log(score_D + eps) \

        # loss_2 = w * (1 - imgs_label) * (score_R + 10 * score_D) * torch.log(gamma - gamma * score_D + eps)
        # loss_2 =   w * (1 - imgs_label) * (score_R + 1.0 + eps + 0 * torch.log(score_D + eps)) * torch.log(gamma * score_D + eps)


        loss_max_d = - loss
        optimizer_D.zero_grad()
        loss_max_d.backward(retain_graph=True)
        optimizer_D.step()


        # ---------------------
        #  Train Recognizer
        # ---------------------
        loss_r = loss
        optimizer_R.zero_grad()

        loss_r.backward()
        optimizer_R.step()

    if opt.model == 'agan':
        baseline = torch.mean(score_D).data
        # score_R.detach_()
        w = 1.0 - torch.exp(- beta * (score_D - 1.0/gamma) ** 4)
        w.detach_()

        def cal_co_rl(epoch_):
            if epoch_ > 10:
                return 0.1
            else:
                return 0.0

        rl_loss_rate = cal_co_rl(epoch_num)
        
        rl_prob, action_index = rl(score_R)
        normal_reward = baseline - score_D
        reverse_reward = score_D - baseline
        
        reward = torch.cat((normal_reward, reverse_reward),1)
        reward = torch.sum(reward * action_index , 1).unsqueeze(1)

        # pdb.set_trace()
        #rl_loss = torch.sum( rl_prob *  torch.log(gamma - gamma * score_D + eps) * rl_loss_rate ) 
        rl_loss = torch.sum( rl_prob * reward)  * rl_loss_rate 
        ptd_loss = torch.sum(imgs_label * torch.log(score_D + eps)) + torch.sum( (1-imgs_label[:positive_num]) * torch.log(1-score_D[:positive_num] + eps))
        #ptd_loss = torch.mean( (1-imgs_label) * torch.log(1-score_D + eps) + imgs_label * torch.log(score_D + eps))
        #print(ptd_loss.size(),rl_loss.size(),type(rl_loss))
        loss = ptd_loss + rl_loss
        loss_max_d = - loss
        if epoch_num < 2000:
            optimizer_D.zero_grad()
            loss_max_d.backward(retain_graph=True)
            optimizer_D.step()


        # ---------------------
        #  Train Recognizer
        # ---------------------

        optimizer_R.zero_grad()

        # loss *= 0.001

        loss.backward()
        # print(torch.sum(list(recognizer.parameters())[0].grad))
        optimizer_R.step()

    if training_iterator._finished_epoch != epoch:
        epoch_num += 1
        #adjust_lr(optimizer_D, epoch_num)
        #adjust_lr(optimizer_R, epoch_num)
        epoch = training_iterator._finished_epoch
        i = 0
        testiterator = pu_dataset.get_testing_iterator()
        correct = 0.0
        all = 0.0
        recognizer.eval()
        # discriminator.eval()
        TP = 0.0
        FP = 0.0
        FN = 0.0
        for test_imgs in testiterator:
            test_imgs_data = Variable(Tensor(test_imgs[0]),requires_grad=False)
            test_imgs_label = Variable(TensorB(test_imgs[1]), requires_grad=False)
            # score = discriminator.forward(test_imgs_data)
            score = recognizer.forward(test_imgs_data)
            #pdb.set_trace()
            label_predicted = score.gt(0.5).byte() != test_imgs_label
            TP += torch.sum(torch.Tensor.float((score[test_imgs_label == 1] >= 0.5).data))
            FP += torch.sum(torch.Tensor.float((score[test_imgs_label == 0] >= 0.5).data))
            FN += torch.sum(torch.Tensor.float((score[test_imgs_label == 1] <= 0.5).data))
            # all_neg_score.append(score[test_imgs_label < 0.5].cpu().data.numpy())

            label_predicted = torch.Tensor.float(label_predicted.data)
            correct += torch.sum(label_predicted)
            all += float(label_predicted.size(0))
        recognizer.train()

        precision = TP / (TP + FP + 1)
        recall = TP / (TP + FN + 1)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        # acc = corrects / eval_num

        #####################################################

            # a , b = torch.max(score,1)
        acc = correct / all
        #print(acc)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_acc.append(acc)

        print("[Epoch %d/%d] [D loss: %f] [Error rate: %f] [precision: %f] [recall: %f] [F1: %f]" % (training_iterator._finished_epoch, opt.n_epochs,
                loss_max_d, acc,precision, recall, f1))
        if acc < big_acc:
            big_acc = acc
        if f1 > big_f:
            big_f = f1
    # batches_done = training_iterator._finished_epoch * training_iterator._len + i
    i=i+1
    # if batches_done % opt.sample_interval == 0:
print("acc")
print(big_acc)
print("f")
print(big_f)    
#     save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
result = {'precision': all_precision,
          'recall': all_recall,
          'f1': all_f1,
          'acc': all_acc}
torch.save(result, opt.dataset + opt.model + '.pkl')



