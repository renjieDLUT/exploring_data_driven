from my_vit import *
from cnn_model import *
if __name__ == "__main__":
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    #  =================================== data ================================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomVerticalFlip(),
         torchvision.transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(norm_mean, norm_std)])
    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(norm_mean, norm_std)])
    train_dataset = torchvision.datasets.CIFAR10(
        './data/cifar10', train=True, transform=train_transform, download=True)

    test_dataset = torchvision.datasets.CIFAR10(
        './data/cifar10', train=False, transform=test_transform, download=True)

    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=6)

    # =================================== model =========================
    # net = MyVIT(image_size=32, patch_size=4, hidden_dim=92, mlp_dim=192, num_classes=10, num_layer=8, num_heads=1,
    #             dropout=0.0, attention_dropout=0.0)
    net=MyDepthWiseModule()
    print(f"train_dataset.shape: {train_dataset[0][0].shape}")
    param_count = 0
    for name, param in net.named_parameters():
        param_count += param.numel()
        # print(name, param.shape)
    print("net param count:{:,d}".format(param_count))

    # ======================================== loss function =======================================
    loss_fn = torch.nn.CrossEntropyLoss()

    # ========================================= optimizer =============================================
    LR = 0.1
    MOMENTA = 0.9
    lr_schedular_step = 20
    lr_decay_gamma = 0.8

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTA)
    lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=lr_decay_gamma)

    # ============================================= 训练 ======================================
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # device=torch.device("cpu")
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    net.to(device)

    for epoch in range(1, epochs + 1):
        loss_value, count = 0., 0
        time_tic = time.time()
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_hat: torch.Tensor = net(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value += loss.item()
            count += (y_hat.argmax(dim=1) == y).sum()
        lr_schedular.step()
        time_toc = time.time()
        print("epoch:{}   loss:{}    acurracy:{}    cost time:".format(epoch, loss_value / train_size,
                                                                       count / train_size),
              time_toc - time_tic)

        loss_value, count = 0., 0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            y_hat: torch.Tensor = net(x)
            loss = loss_fn(y_hat, y)

            loss_value += loss.item()
            count += (y_hat.argmax(dim=1) == y).sum()

        print("epoch:{}    loss:{}    acurracy:{}".format(epoch, loss_value / test_size,
                                                          count / test_size))
