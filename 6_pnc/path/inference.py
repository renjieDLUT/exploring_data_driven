import matplotlib.pyplot as plt
import numpy as np

from path_transformer import PathTransformer
from generate_world import *
import torch

model_pt_path = './tmp/path_transformer_60.pt'

checkpoint = torch.load(model_pt_path)

obstacles_mean = checkpoint["obstacles_mean"]
obstacles_std = checkpoint["obstacles_std"]
ref_line_points_mean = checkpoint["ref_line_points_mean"]
ref_line_points_std = checkpoint["ref_line_points_std"]
planned_discrete_points_mean = checkpoint["planned_discrete_points_mean"]
planned_discrete_points_std = checkpoint["planned_discrete_points_std"]
planned_discrete_sl_points_mean = checkpoint["planned_discrete_sl_points_mean"]
planned_discrete_sl_points_std = checkpoint["planned_discrete_sl_points_std"]

ref_line_cubic_poly = generate_cubic_poly_wrt_x()
ref_line_length = 100.0
# ref_line_input = list(ref_line_cubic_poly.coef) + [ref_line_length]
ref_line_discrete_path = convert_path(ref_line_cubic_poly, max_x=ref_line_length)
ref_line_discrete_points = ref_line_discrete_path.get_points(100.0, 201)
obstacle_list = generate_obs_along_path(ref_line_discrete_path, max_s=100.0)

obstacle_list_input = (obstacle_list - obstacles_mean) / obstacles_std
ref_line_input = (ref_line_discrete_points - ref_line_points_mean) / ref_line_points_std

obstacle_list_input = torch.tensor(obstacle_list_input).unsqueeze(dim=0).to(torch.float32)
ref_line_input = torch.tensor(ref_line_input).unsqueeze(dim=0).to(torch.float32)

loc=ref_line_discrete_points[0]
loc_input=(loc-planned_discrete_points_mean)/planned_discrete_points_std
loc_input=torch.tensor(loc_input).unsqueeze(dim=0).to(torch.float32)

model = checkpoint['model']
model.to("cpu")
model.eval()
pred = model(obstacle_list_input, ref_line_input, loc_input)

# pred_path = pred[0].detach().numpy() * planned_discrete_sl_points_std + planned_discrete_sl_points_mean

pred_path = pred[0].detach().numpy() * planned_discrete_points_std + planned_discrete_points_mean

# xs = []
# ys = []
# for sl in pred_path:
#     x, y = ref_line_discrete_path.get_xy(sl[0], sl[1])
#     print(sl[0],end=" ")
#     xs.append(x)
#     ys.append(y)

planned_path = DiscretePath(list(pred_path[:, 0]), list(pred_path[:, 1]))

fig, ax = plt.subplots()
ref_line_discrete_path.show(ax)
plot_obs(ax, obstacle_list)
planned_path.show(ax, color='r')
plt.show()
