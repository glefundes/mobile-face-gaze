import os
import time
import utils
import torch
import argparse
import torch.nn as nn
from models import gazenet
from mpiifacegaze_dataset.dataloader import get_loader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# Argparser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='/dataset/raw/')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--log_path', type=str, default='training.log')
parser.add_argument('--output', '-o', type=str, default='.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--lr_decay', type=float, default=0.1)
args = parser.parse_args()

# Logging
args.log_path = os.path.join(args.output, args.log_path)
log = []
log.append('Learning Rate: %s' % args.learning_rate)
log.append('Batch Size: %s' % args.batch_size)
log.append('Log File Path: %s' % args.log_path)
with open(args.log_path, 'w') as f:
    for line in log:
        print(line)
        f.write('{}\n'.format(line))


# Loss function 
loss_fn = nn.L1Loss(reduction='mean')

# Model instance, optimizer and LR Scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = gazenet.GazeNet(device=device).train()
optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8], gamma=args.lr_decay)


# Dataset
train_loader = get_loader(args.dataset, args.batch_size)

#Training Loop
start_time = time.time() 
print('Training started at {}'.format(time.strftime("%a, %d %b %Y %H:%M:%S ", time.gmtime())))

for epoch in range(args.epochs):
    for batch_idx, (imgs, gt) in enumerate(train_loader):

        imgs = imgs.float().cuda()
        gt = gt.cuda()

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = loss_fn(outputs, gt)
        loss.backward()

        optimizer.step()
        angle_error = utils.compute_angle_error(outputs, gt).mean()

        if batch_idx % 100 == 0:
            s = ('Epoch {} Step {}/{} '
                        'Loss: {:.4f} '
                        'AngleError: {:.2f}'.format(
                            epoch,
                            batch_idx,
                            len(train_loader),
                            loss.item(),
                            angle_error.item(),
                        ))
            print(s)
            with open(args.log_path, 'a') as f:
                f.write('{}\n'.format(s))

    
    print('epoch finished')
    elapsed = time.time() - start_time
    print('Elapsed {:.2f} min'.format(elapsed/60))
    print('===================================')
    torch.save(model.state_dict(), os.path.join(args.output, 'model-{}.pth'.format(epoch)))

