import os.path
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json


class PathDataset(Dataset):
    def __init__(self, file_dir):
        super().__init__()
        files = [os.path.join(file_dir, path) for path in os.listdir(file_dir) if
                 os.path.isfile(os.path.join(file_dir, path))]
        files = [file for file in files if os.path.splitext(file)[1] == '.json']

        self.data_info = []
        for file in files:
            with open(file, 'r+') as fb:
                self.data_info.extend(json.load(fb))
        self.obstacles, self.ref_line, self.ref_line_points, self.planned_sl_path, self.planned_discrete_points, self.planned_discrete_sl_points = self.get_info()

        self.obstacles_mean, self.obstacles_std = self.calculate_obstacles_mean_std()

        # self.ref_line_mean = np.mean(self.ref_line, 0)
        # self.ref_line_std = np.std(self.ref_line, 0)

        self.ref_line_points_mean = np.mean(self.ref_line_points, axis=(0, 1))
        self.ref_line_points_std = np.std(self.ref_line_points, axis=(0, 1))

        # self.planned_sl_path_mean = np.mean(self.planned_sl_path, 0)
        # self.planned_sl_path_std = np.std(self.planned_sl_path, 0)

        self.planned_discrete_points_mean = np.mean(self.planned_discrete_points, axis=(0, 1))
        self.planned_discrete_points_std = np.std(self.planned_discrete_points, axis=(0, 1))

        # self.planned_discrete_sl_points_mean = np.mean(self.planned_discrete_sl_points, axis=(0, 1))
        # self.planned_discrete_sl_points_std = np.std(self.planned_discrete_sl_points, axis=(0, 1))

        self.relative_point = np.asarray(self.ref_line_points)[:, 0, :]

    def get_info(self):
        obs = []
        ref_line = []
        ref_line_points = []
        planned_sl_path = []
        planned_discrete_points = []
        planned_discrete_sl_points = []
        for data in self.data_info:
            if data['obstacles']:
                obs.append(data["obstacles"])
            else:
                obs.append([[0.0] * 5])
            ref_line.append(data["centric_cubic_poly"]["coeff"] +
                            [data["centric_cubic_poly"]["centric_line_max_x"]])
            ref_line_points.append(data["centric_line_discrete_point"])
            if (not data["planned_sl_path"]["coeff"]) or (not data["planned_sl_path"]["max_s"]):
                planned_sl_path.append([0.0] * 5)
            else:
                planned_sl_path.append(
                    data["planned_sl_path"]["coeff"] + [data["planned_sl_path"]["max_s"]])
            planned_discrete_points.append(data['planned_discrete_points'])
            planned_discrete_sl_points.append(data['planned_discrete_sl_points'])
        return obs, ref_line, ref_line_points, planned_sl_path, planned_discrete_points, planned_discrete_sl_points

    def calculate_obstacles_mean_std(self):
        all_obstacles = []
        for obstacles in self.obstacles:
            all_obstacles.extend(obstacles)
        return np.mean(all_obstacles, 0), np.std(all_obstacles, 0)

    @staticmethod
    def get_obs_point(obs_box):
        c_x, c_y, w, h, heading = obs_box
        line1 = (math.cos(heading) * w / 2, math.sin(heading) * w / 2)
        line2 = (math.cos(heading + math.pi / 2) * h / 2, math.sin(heading + math.pi / 2) * h / 2)
        p1 = [c_x + line1[0] + line2[0], c_y + line1[1] + line2[1]]
        p2 = [c_x - line1[0] + line2[0], c_y - line1[1] + line2[1]]
        p3 = [c_x - line1[0] - line2[0], c_y - line1[1] - line2[1]]
        p4 = [c_x + line1[0] - line2[0], c_y + line1[1] - line2[1]]
        rec_points = [p1, p2, p3, p4, p1]
        return rec_points

    def __getitem__(self, item):
        relative_point = self.relative_point[item]

        obstacle = self.obstacles[item]
        obs_points = []
        for obs in obstacle:
            obs_points.append(self.get_obs_point(obs))

        obstacle = np.asarray(obstacle)
        obstacle[:, :2] = obstacle[:, :2] - relative_point
        normed_obstacle = (obstacle - self.obstacles_mean) / self.obstacles_std

        ref_line_points = self.ref_line_points[item]
        normed_ref_line_points: np.ndarray = ((ref_line_points - relative_point - self.ref_line_points_mean)
                                              / self.ref_line_points_std)

        normed_obs_points: np.ndarray = ((obs_points - relative_point - self.ref_line_points_mean)
                                         / self.ref_line_points_std)
        normed_obs_points = normed_obs_points.reshape(-1, 10)

        planned_discrete_points = self.planned_discrete_points[item]
        normed_planned_discrete_points: np.ndarray = (
                (planned_discrete_points - relative_point - self.planned_discrete_points_mean)
                / self.planned_discrete_points_std)

        location = normed_planned_discrete_points[0, :]

        return normed_obstacle.astype(np.float32), normed_obs_points.astype(np.float32), normed_ref_line_points.astype(
            np.float32), normed_planned_discrete_points.astype(np.float32), location.astype(np.float32)

    def __len__(self):
        return len(self.data_info)


# def path_collate(data)

if __name__ == "__main__":
    file_path = './data/pnc_path/cubic_poly/train'
    path_dataset = PathDataset(file_path)
    for i in range(100):
        normed_obstacle, normed_obs_points, normed_ref_line, normed_planned_discrete_points, normed_loc = path_dataset[
            i]
        print(type(normed_obstacle), type(normed_obs_points), type(normed_ref_line),
              type(normed_planned_discrete_points), type(normed_loc))
        # print(normed_loc*path_dataset.planned_discrete_points_std+path_dataset.planned_discrete_points_mean)
        print(normed_obs_points.shape)
        obs = normed_obstacle * path_dataset.obstacles_std + path_dataset.obstacles_mean
        ref_line = normed_ref_line * path_dataset.ref_line_points_std + path_dataset.ref_line_points_mean
        planned_discrete_points = normed_planned_discrete_points * path_dataset.planned_discrete_points_std + path_dataset.planned_discrete_points_mean

        from generate_world import *

        fig, ax = plt.subplots()
        plot_obs(ax, obs)
        ref_line_x, ref_line_y = ref_line[:, 0], ref_line[:, 1]
        ref_line_path = DiscretePath(list(ref_line_x), list(ref_line_y))
        ref_line_path.show(ax)
        # xs = []
        # ys = []
        # for sl in sl_points:
        #     x, y = ref_line_path.get_xy(sl[0], sl[1])
        #     xs.append(x)
        #     ys.append(y)
        # print(xs,ys)
        planned_path = DiscretePath(list(planned_discrete_points[:, 0]), list(planned_discrete_points[:, 1]))
        planned_path.show(ax, color="r")
        plt.show()

        dataloader = DataLoader(path_dataset, batch_size=1)

        obs,obs_points, ref_line, planned_discrete_points,loc = next(iter(dataloader))
        print(obs_points.shape, ref_line.shape, planned_discrete_points.shape,loc.shape)
