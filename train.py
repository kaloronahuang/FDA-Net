# train.py
import os, json, torch, trainer
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from models import FDANetGenerator, FDANetDiscriminator
from torch.utils.data import DataLoader
from dataset import FDAImageDataset
from torchvision.transforms import ToTensor
from torch import optim

cfg = json.load(open('./config.json', 'r'))
embed_net = InceptionResnetV1(pretrained='vggface2').eval()
netG = FDANetGenerator(cfg['embedding_size'], cfg['feature_size'])
netD = FDANetDiscriminator(cfg['feature_size'])

if input('Load from existing models? (y/n)')[0:1].lower() == 'y':
    ckpt_list = os.listdir(cfg['checkpoint_folder'])
    print('Existing checkpoints are listed below:')
    for dname in ckpt_list:
        print(f'\t{dname}')
    tag = input('Please input the checkpoint tag: ')
    netG.load_state_dict(torch.load(os.path.join(cfg['checkpoint_folder'], f'model-{tag}', 'netG.pth')))
    netD.load_state_dict(torch.load(os.path.join(cfg['checkpoint_folder'], f'model-{tag}', 'netD.pth')))
else:
    netG.apply(trainer.weight_init)
    netD.apply(trainer.weight_init)

print('Models are loaded.')

if cfg['device'] == 'cuda':
    print('Using GPU Acceleration...')
    if cfg['ngpu'] == 0:
        print('Invalid Config: ngpu must greater than zero.')
        exit(0)
    netG.to(cfg['device'])
    netD.to(cfg['device'])
    embed_net.to(cfg['device'])

    if cfg['ngpu'] > 1:
        netG = nn.DataParallel(netG, list(range(cfg['ngpu'])))
        netD = nn.DataParallel(netD, list(range(cfg['ngpu'])))
        print('Multi-GPU enabled.')
    print('Moved to GPU.')

dataloader = DataLoader(
    FDAImageDataset(
        cfg['embedding_path'],
        cfg['image_dir'],
        transform=ToTensor()
    ),
    batch_size=cfg['batch_size'],
    shuffle=True
)

netD_criterion = nn.BCELoss()
dist_criterion = nn.L1Loss()
optimG = optim.Adam(netG.parameters(), lr=cfg['learning_rate'], betas=cfg['betas'])
optimD = optim.Adam(netD.parameters(), lr=cfg['learning_rate'], betas=cfg['betas'])

history = {
    'current_epoch': -1,
    'epochs': []
}
if os.path.exists(cfg['history_path']):
    history = json.load(open(cfg['history_path'], 'r'))

while True:
    history['current_epoch'] += 1
    epoch_hist = trainer.train_batch(cfg, dataloader, netG, netD, embed_net, optimG, optimD, netD_criterion, dist_criterion, history['current_epoch'])
    history['epochs'].append(epoch_hist)
    json.dump(history, open('history.json', 'w'))