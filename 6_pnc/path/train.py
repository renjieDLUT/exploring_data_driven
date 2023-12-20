import torch.optim
import torch.nn as nn
from path_transformer import PathTransformer
from path_dataset import PathDataset

from torch.utils.data import DataLoader

file_dir = './data/pnc_path/cubic_poly/train'
path_dataset = PathDataset(file_dir)
path_test_dataset = PathDataset('./data/pnc_path/cubic_poly/test')

dataloader = DataLoader(path_dataset, batch_size=1)
test_dataloader = DataLoader(path_test_dataset, batch_size=1)

model = PathTransformer()
# model_pt_path = './tmp/path_transformer_10.pt'
# checkpoint = torch.load(model_pt_path)
# model = checkpoint['model']

device = 'cuda' if torch.cuda.is_available() else "cpu"
model.to(device=device)

count = 0
for name, param in model.named_parameters():
    count += param.numel()
print(f'total param number:{count:,d}')

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.8)

loss_fn = nn.L1Loss()

epoch = 1000
for epo in range(epoch):
    sum_loss = 0.0
    iter_sum_loss = 0.0
    for iter, (obs,obs_points, ref_line, y, loc) in enumerate(dataloader):
        obs_points = obs_points.to(device=device)
        ref_line = ref_line.to(device=device)
        loc = loc.to(device=device)
        y = y.to(device=device)

        y_hat = model(obs_points, ref_line, loc)
        l = loss_fn(y_hat, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        sum_loss += l.item()
        iter_sum_loss += l.item()
        if (iter + 1) % 100 == 0:
            # print(f'iteration:{iter} loss:{iter_sum_loss / 100}')
            iter_sum_loss = 0.0

    if (epo + 1) % 10 == 0:
        model_pt_path = f'./tmp/path_transformer_{epo + 1}.pt'
        torch.save({"model": model,
                    "obstacles_mean": path_dataset.obstacles_mean,
                    "obstacles_std": path_dataset.obstacles_std,
                    "ref_line_points_mean": path_dataset.ref_line_points_mean,
                    "ref_line_points_std": path_dataset.ref_line_points_std,
                    "planned_discrete_points_mean": path_dataset.planned_discrete_points_mean,
                    "planned_discrete_points_std": path_dataset.planned_discrete_points_std,
                    }, model_pt_path)

    lr_schedular.step()
    print(f'epoch:{epo}  loss:{sum_loss / len(dataloader)}')

    model.eval()
    sum_test_loss = 0.0
    with torch.no_grad():
        for obs,obs_points, ref_line, y, loc in test_dataloader:
            obs_points = obs_points.to(device=device)
            ref_line = ref_line.to(device=device)
            loc = loc.to(device=device)
            y = y.to(device=device)

            y_hat = model(obs_points, ref_line, loc)
            l = loss_fn(y_hat, y)
            sum_test_loss += l.item()

    print('*' * 20 + f'test loss:{sum_test_loss / len(test_dataloader)}' + '*' * 20)
    model.train()
