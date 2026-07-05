import os, sys, argparse
import onnx
import onnxruntime
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                   ])

training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 64
train_dataloader = DataLoader(training_data, shuffle=True,
                            batch_size=batch_size, num_workers=2)
test_dataloader = DataLoader(test_data, shuffle=False,
                            batch_size=batch_size, num_workers=2)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        feat = self.conv(x)
        output = self.fc(feat.view(x.shape[0], -1))
        return output

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

pth_file = "model.pth"

def run_train(args):
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Training Done!")

    torch.save(model.state_dict(), pth_file)
    print("Saved PyTorch Model State to:", pth_file)

from d2l import torch as d2l
from PIL import Image
import cv2

def run_test(args):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(pth_file))
    model.eval()
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",]

    with torch.no_grad():
        if (args.i):
            print("input jpg:", args.i)
            img = cv2.imread(args.i, cv2.IMREAD_GRAYSCALE)
            #img = Image.open(args.i)
            #img = torchvision.io.read_image(args.i, torchvision.io.image.ImageReadMode.GRAY)
            x = transform(img)
            # unsqueeze(0): Fix RuntimeError: Expected 4-dimensional input for 4-dimensional weight [6, 1, 5, 5],
            # but got 3-dimensional input of size [1, 28, 28] instead
            x = x.unsqueeze(0)
            x = x.to(device)
            print("x.shape:", x.shape)
        else:
            if (args.b):
                print("input binary:", args.b)
                img = np.fromfile(args.b, dtype=np.uint8)
                x = torch.Tensor(img.reshape(1, 28, 28))  ## NO preprocess
                #x = transform(img.reshape(28, 28))    ## Yes preprocess
                print("x.shape:", x.shape)
            else:
                x, y = test_data[0][0], test_data[0][1]
                actual = classes[y]
                print("Pick the first data from test dataset, actual lable:", actual)
            x = x.unsqueeze(0)
            x = x.to(device)
            print("x.shape:", x.shape)

        pred = model(x)
        predicted, = classes[pred[0].argmax(0)],
        print("scoes:", pred[0])
        print(f'Predicted: "{predicted}"')

def export_to_onnx(args):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(pth_file))
    model.eval()
    input_names = ["input"]
    output_names = ["output"]
    x = torch.randn(1, 1, 28, 28)
    onnx_model = args.onnx
    torch.onnx.export(model, x, onnx_model, export_params=True,
                      verbose = False,
                      input_names=input_names,
                      output_names=output_names)
    print(f"Export PyTorch Model: {pth_file} to ONNX: {onnx_model}")

def main(args):
    if args.train:
        run_train(args)
    elif args.test:
        run_test(args)
    elif args.onnx:
        export_to_onnx(args)
    else:
        print("No action")

def init_param(args):
    parser = argparse.ArgumentParser(description="Train and test Lenet with MNIST dataset")
    parser.add_argument("-train", action='store_true', required=False,
        help="Train model")
    parser.add_argument("-test", action='store_true', required=False,
        help="Test model")
    parser.add_argument("-i", type=str, required=False,
        help="input *.jpg, *.png filename")
    parser.add_argument("-b", type=str, required=False,
        help="input binary filename, after preprocess (like submean, scale)")
    parser.add_argument("-onnx", type=str, required=False,
        help="export to onnx model")

    return parser.parse_args(args)

if __name__ == '__main__':
	args = init_param(sys.argv[1:])
	main(args)
