import random
import torch
from torch import nn, optim
from torch.utils import data
import librosa

# Sample CNN model
#
# https://arxiv.org/abs/1703.01789
#
# As implemented by https://github.com/kyungyunlee/sampleCNN-pytorch
class SampleCNN(nn.Module):
    """SampleCNN model
    See: https://arxiv.org/abs/1703.01789"""
    def __init__(self):
        super(SampleCNN, self).__init__()
        # 59049 x 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 6561 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 2187 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 729 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 243 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3),
            nn.Dropout(0.5))
        # 81 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 27 x 256
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 9 x 256
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 3 x 256
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 1 x 512 
        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5))
        # 1 x 512 
        self.fc = nn.Linear(512, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        x = x.view(x.shape[0], 1, -1)
        # x : 23 x 1 x 59049
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        out = self.fc(out)
        out = self.activation(out)
        return out

class Dataset(data.IterableDataset):
    """Random window sampler.
    
    This class yields randomly-selected windows of the given length from the data
    pool."""
    def __init__(self, positive, negative, window=59049, rate=16000, load=librosa.core.load):
        super(Dataset, self).__init__()
        self.data = ([load(f, rate) for f in negative], [load(f, rate) for f in positive])
        self.window = window
    def __iter__(self):
        while True:
            # Choose from random negative (0) or positive (1) sample pool? 
            k = random.randrange(2)
            # Choose random sample from selected pool
            data, sr = random.choice(self.data[k])
            # Choose random window from selected sample.
            i = random.randrange(self.window, len(data)) - self.window
            # Yield pool index {0, 1} and sample window.
            yield data[i:i+self.window], k

def trainer(net, criterion, optimizer, scheduler):
    """Training loop."""
    result = None
    while True:
        net.zero_grad()
        X, y = (yield result)
        loss = criterion(net(X).flatten(), y.to(torch.float))
        result = float(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

if __name__ == "__main__":
    import os
    import math
    #
    # Constants
    #
    MODEL_PATH = "model-gztan-speech-music.pth"
    OPTIM_PATH = "opt-gztan-speech-music.pth"
    SCHED_PATH = "sched-gztan-speech-music.pth"
    COUNT_PATH = "batches-gztan-speech-music.txt"
    #
    # Iteration limit.
    #
    try:
        iterations = int(os.environ["LIMIT"])
    except KeyError:
        iterations = math.inf
    #
    # Use GPU if available.
    #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    #
    # Create model, load saved state if available.
    #
    net = SampleCNN()
    try:
        net.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        pass
    #
    # Create optimizer, load saved state if available.
    #
    optimizer = optim.SGD(params=net.parameters(), lr=0.01)
    try:
        optimizer.load_state_dict(torch.load(OPTIM_PATH))
    except FileNotFoundError:
        pass
    #
    # Create cyclic learning rate scheduler, load saved state if available. 
    #
    scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-8, max_lr=0.01)
    try:
        scheduler.load_state_dict(torch.load(SCHED_PATH))
    except FileNotFoundError:
        pass
    #
    # Load current batch count from file, or set to 0 if we're training
    # from scratch.
    #
    try:
        with open(COUNT_PATH, "r") as fobj:
            i = fobj.read()
            i = i.strip()
            if i:
                i = int(i)
    except (FileNotFoundError, ValueError):
        i = 0
    #
    # Move model to device.
    #
    net = net.to(device)
    #
    # Load dataset, create loader.
    #
    with os.scandir("music_speech/music_wav") as negdir, os.scandir("music_speech/speech_wav") as posdir:
        dataset = Dataset((f.path for f in posdir), (f.path for f in negdir))
    loader = data.DataLoader(dataset, batch_size=32)
    #
    # Binary cross-entropy loss.
    #
    criterion = nn.BCELoss()
    #
    # Instantiate trainer coroutine.
    #
    tr = trainer(net, criterion, optimizer, scheduler)
    tr.send(None)
    #
    # Iterate through batches.
    #
    for i, Xy in enumerate(loader, i + 1):
        try:
            loss = tr.send(Xy)
            print("Batch {}: Loss = {}".format(i, loss))
            # Save training state at the end of every CyclicLR cycle.
            if i % 4000 == 0:
                torch.save(net.state_dict(), MODEL_PATH)
                torch.save(optimizer.state_dict(), OPTIM_PATH)
                torch.save(scheduler.state_dict(), SCHED_PATH)
        except KeyboardInterrupt:
            break
        if i >= iterations:
            break
    #
    # Save batch index and current training state.
    #
    with open(COUNT_PATH, "w") as fobj:
        fobj.write(str(i))
    torch.save(net.state_dict(), MODEL_PATH)
    torch.save(optimizer.state_dict(), OPTIM_PATH)
    torch.save(scheduler.state_dict(), SCHED_PATH)
