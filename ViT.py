from ViT.utils.utilities import *


BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 16


DL_PATH = "C:\Pytorch\Spyder\CIFAR10_data" # Use your own path
# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
transform = torchvision.transforms.Compose(
     [#torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),   #Augment
     #torchvision.transforms.RandomAffine(8, translate=(.15,.15)),    #Augment
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#train_dataset = datasets.ImageFolder('custom dataset',transform=transform)
#test_dataset = datasets.ImageFolder('custom dataset',transform=transform)
#train_size = math.floor(len(dataset)*0.8)
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = data.random_split(dataset,lengths=[train_size,test_size])
train_dataset = torchvision.datasets.CIFAR10(DL_PATH, train=True,download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(DL_PATH, train=False,download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST,
                                         shuffle=False)
device=device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")



N_EPOCHS = 150

model = ImageTransformer(image_size=32, patch_size=4, num_classes=32, channels=3,
            dim=64, depth=6, heads=8, mlp_dim=128)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,1,verbose=False)


if __name__=='main':
    print('Epoch:', epoch)
    start_time = time.time()
    model=train(train_loader, test_loader, model, criterion, optimizer, exp_lr_scheduler, N_EPOCHS)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    print(test_acc(model, test_loader))

print('Execution time')
