# trainer.py
import os, types, torch
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
from models import FDANetGenerator, FDANetDiscriminator
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from PIL import Image

import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_sample(config: dict, additional_text: str, generated_images: torch.Tensor):
    plt.figure(figsize=(config['preview_image']['nrow'], config['preview_image']['ncol']))
    plt.axis('off')
    plt.title('Sample' if additional_text == '' else f'Sample - {additional_text}')
    plt.imshow(np.transpose(vutils.make_grid(generated_images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(config['preview_image']['path'], format='eps')

def save_model(config: dict, tag: str, netG: nn.Module, netD: nn.Module):
    model_path = os.path.join(config['checkpoint_folder'], f'model-{tag}')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(netG.state_dict(), os.path.join(model_path, 'netG.pth'))
    torch.save(netD.state_dict(), os.path.join(model_path, 'netD.pth'))

def train_batch(
        config: dict,
        dataloader: torch.utils.data.DataLoader,
        netG: nn.Module,
        netD: nn.Module,
        embed_net: nn.Module,
        optimG: optim.Optimizer,
        optimD: optim.Optimizer,
        lossDiscriminator,
        lossDist,
        epochId = 0
    ):
    real_label = Variable(torch.full((config['batch_size'], ), 1.0, dtype=torch.float, device=config['device']), requires_grad=False)
    fake_label = Variable(torch.full((config['batch_size'], ), 0.0, dtype=torch.float, device=config['device']), requires_grad=False)
    training_history = {
        'epoch_id': epochId,
        'history': {
            'errD': [],
            'errG': []
        },
        'niteration': 0,
        'interval': 10
    }
    for batch_id, data in enumerate(dataloader, 0):
        imgs = data['imgs'].to(config['device'])
        embeds = data['embeddings'].to(config['device'])

        netD.zero_grad()
        pred = netD(imgs).view(-1)
        errD_real = lossDiscriminator(pred, real_label)
        
        fake = netG(embeds)
        pred = netD(fake.detach()).view(-1)
        errD_fake = lossDiscriminator(pred, fake_label)

        errD = (errD_real + errD_fake)
        errD.backward()
        optimD.step()

        netG.zero_grad()
        pred = netD(fake).view(-1)
        errG_genuity = lossDiscriminator(pred, real_label)

        pred = embed_net(fake)
        errG_relate = lossDist(pred, embeds) / (config['embedding_size']**(0.5))

        errG = (errG_genuity + errG_relate)
        errG.backward()
        optimG.step()
        
        training_history['niteration'] += 1
        if batch_id % 10 == 0:
            training_history['history']['errG'].append(errG.cpu().item())
            training_history['history']['errD'].append(errD.cpu().item())

            print('[Epoch %d][Batch %d]errG: %.4f, errD: %.4f' % (epochId, batch_id, errG.cpu().item(), errD.cpu().item()))