import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#The main components of this code are from Shake's demonstration

#Creates a Main function that loads the data and runs the model
if __name__ == '__main__':
    #Sets torch to device and uses GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Trasforms the data by: Normalising with values from the internet 
    #Random horizontal fliping and random corpp
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.44465), (0.2023,0.1994,0.2010)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    ])

    #Transforms the test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #Divides the train set into batches and shuffles the data such that there is 
    trainset = torchvision.datasets.CIFAR10(root='cifar10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=6)

    #Divides the test set into batches
    testset = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=6)
    
    #Creates the Basicblock 
    class BasicBlock(nn.Module):
        #Contolls the number of chanels the internal layer will have 
        expansion = 1


        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            #Creates 2 convolutional layers and applies batch normalisation
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            #Skips teh connection and serves as the identity if needed 
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        #Forward prop that runs infference
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            #Does nothing if the connection is skipped
            out += self.shortcut(x)
            out = F.relu(out)
            return out 
        

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            #number of feature maps/chanels
            self.in_planes = 64

            #Initial convolutional lyer
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)

            #Explains the number of residual blocks at each layer 
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)


        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        #We have 4 distict groups of layers 
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
        
    #Creating the ResNetStricture consisting of 8 basic blocks 
    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])

    model = ResNet18()
    model = model.to(device)

    #Uses Stocastic Gradient Decent 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.1, 
                                momentum=0.9, 
                                weight_decay=5e-4)

    # Piecewise Linear Schedule which increases the learning rate then decreases it
    total_step = len(train_loader)
    #Increases learning rate in 15 steps from 0.005 to 0.1
    #Decreases learnign rate in 15 steps form 0.1 to 0.005
    sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=0.005, 
        max_lr=0.1, 
        step_size_up=15, 
        step_size_down=15, 
        mode="triangular")
    
    #Linearly decreases the learning rate 
    sched_linear_3 = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.005/0.1, 
        end_factor=0.005/5, 
        verbose=False)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[sched_linear_1, 
                    sched_linear_3], 
                    milestones=[30])

    num_epochs = 35

    model.train()
    print("> Training")
    

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        scheduler.step()

    #Testing the model 
    print("> Testing")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))