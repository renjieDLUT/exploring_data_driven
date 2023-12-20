import matplotlib.pyplot as plt
import numpy as np

from path_transformer import PathTransformer
from generate_world import *
from path_dataset import PathDataset
import torch

model_pt_path = './tmp/path_transformer_10.pt'

checkpoint = torch.load(model_pt_path)

obstacles_mean = checkpoint["obstacles_mean"]
obstacles_std = checkpoint["obstacles_std"]
ref_line_points_mean = checkpoint["ref_line_points_mean"]
ref_line_points_std = checkpoint["ref_line_points_std"]
planned_discrete_points_mean = checkpoint["planned_discrete_points_mean"]
planned_discrete_points_std = checkpoint["planned_discrete_points_std"]


file_path = './data/pnc_path/cubic_poly/train'
path_dataset = PathDataset(file_path)

while True:
    index = random.randint(0, len(path_dataset))
    obs,obs_points, ref_line, planned_points, loc = path_dataset[index]
    relative_point = path_dataset.relative_point[index]

    obs_input = torch.tensor(obs).unsqueeze(dim=0).to(torch.float32)
    obs_points_input = torch.tensor(obs_points).unsqueeze(dim=0).to(torch.float32)
    ref_line_input = torch.tensor(ref_line).unsqueeze(dim=0).to(torch.float32)
    loc_input = torch.tensor(loc).unsqueeze(dim=0).to(torch.float32)

    model = checkpoint['model']
    model.to("cpu")
    model.eval()
    pred = model(obs_points_input, ref_line_input, loc_input)


    obs = obs * path_dataset.obstacles_std + path_dataset.obstacles_mean
    obs[:,:2]+= relative_point
    ref_line = ref_line * path_dataset.ref_line_points_std + path_dataset.ref_line_points_mean + relative_point
    planned_discrete_points = planned_points * path_dataset.planned_discrete_points_std + path_dataset.planned_discrete_points_mean + relative_point
    loc = loc * path_dataset.planned_discrete_points_std + path_dataset.planned_discrete_points_mean + relative_point
    pred_path = pred[0].detach().numpy() * planned_discrete_points_std + planned_discrete_points_mean + relative_point

    ref_line_path = DiscretePath(list(ref_line[:, 0]), list(ref_line[:, 1]))
    planned_path = DiscretePath(list(pred_path[:, 0]), list(pred_path[:, 1]))
    label_path = DiscretePath(list(planned_discrete_points[:, 0]), list(planned_discrete_points[:, 1]))

    fig, ax = plt.subplots()
    ref_line_path.show(ax)
    planned_path.show(ax, color='g')
    plot_obs(ax, obs)
    label_path.show(ax, color="r")
    ax.plot(loc[0], loc[1], "ro")
    plt.show()
