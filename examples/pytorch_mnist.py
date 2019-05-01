from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd
import tensorboardX
import math

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# ---- args added by jack and yuan
# for some reason, new args aren't working quite right
parser.add_argument('--loadcp', action='store_true', default=False,
                    help='whether or not to try to load a checkpoint before starting training')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')

parser.add_argument('--batches-per-allreduce', type=int, default=1,
                     help='number of batches processed locally before '
                          'executing allreduce across workers; it multiplies '
                          'total batch size.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# yuan add
allreduce_batch_size = args.batch_size * args.batches_per_allreduce

# Horovod: initialize library.
hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs)

test_dataset = \
    datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(),
                      lr=(args.lr * args.batches_per_allreduce *
                          hvd.size()),
                      momentum=args.momentum)

for name, p in model.named_parameters():
    print(name)
    print(p.data.size())

# Horovod: write TensorBoard logs on first worker
log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=args.batches_per_allreduce)

resume_from_epoch = 0
#try to load an existing model if it exists
if args.loadcp and hvd.rank() == 0:
    try:
        checkpoint = torch.load('saved_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_from_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("resuming training from epoch " + str(resume_from_epoch))

    except:
       print("no saved model found")

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

print("before broadcast epoch")
# Horovod: broadcast resume_from_epoch from rank 0 (which will have ckpts)
# to other ranks
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()
print("after broadcast epoch", resume_from_epoch)

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        # get the index of the max log-probability
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
    #if True:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        # loss and accuracy for each batch
        train_batch_loss = 0.
        train_batch_accuracy = 0.

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # split data into sub-batches of size batch_size
        # we are now of allreduce_batch_size
        for i in range(0, len(data), args.batch_size):
            data_batch = data[i:i + args.batch_size]
            target_batch = target[i:i + args.batch_size]
            output = model(data_batch)
            #train_accuracy.update(accuracy(output, target_batch))
            loss = F.nll_loss(output, target_batch)
            # average gradients among sub-batches
            loss.div_(math.ceil(float(len(data)) / args.batch_size))
            loss.backward()

        bigoutput = model(data)

        train_batch_loss = F.nll_loss(bigoutput, target).item()
        train_batch_accuracy = accuracy(bigoutput, target)
        train_batch_loss = metric_average(train_batch_loss, 'avg_train_batch_loss')
        train_batch_accuracy = metric_average(train_batch_accuracy, 'avg_train_batch_accuracy')
        optimizer.step()

        if hvd.rank() == 0 and batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}, Batch Accuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader),
                train_batch_loss, train_batch_accuracy))
        
    #save the model after some number of epochs
    if epoch == 2 and hvd.rank() == 0:
        print("Saving model...")
        save_model(epoch, loss)

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

#def save_model(epoch, loss, batch_size, backward_passes_per_step):
def save_model(epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
#        'batch_size': batch_size,
#        'backward_passes_per_step': backward_passes_per_step
    }, 'saved_model.pt')

if resume_from_epoch > 0:
    test()

for epoch in range(resume_from_epoch+1, args.epochs + 1):
    train(epoch)
    test()

