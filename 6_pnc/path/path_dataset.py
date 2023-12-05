import os.path
import numpy as np
from torch.utils.data import Dataset
import json


class PathDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        _, ext_name = os.path.splitext(file_path)
        assert ext_name == '.json', "file should be json format"
        self.data_info = None
        with open(file_path, 'r+') as fb:
            self.data_info = json.load(fb)
        self.obstacles, self.ref_line, self.planned_sl_path = self.get_info()

        self.obstacles_mean,self.obstacles_std  = self.calculate_obstacles_mean_std()

        self.ref_line_mean = np.mean(self.ref_line, 0)
        self.ref_line_std = np.std(self.ref_line, 0)

        self.planned_sl_path_mean = np.mean(self.planned_sl_path, 0)
        self.planned_sl_path_std = np.std(self.planned_sl_path, 0)

    def get_info(self):
        obs = []
        ref_line = []
        planned_sl_path = []
        for data in self.data_info:
            obs.append(data["obstacles"])
            ref_line.append(data["centric_cubic_poly"]["coeff"] +
                            [data["centric_cubic_poly"]["centric_line_max_x"]])
            if (not data["planned_sl_path"]["coeff"]) or (not data["planned_sl_path"]["max_s"]):
                planned_sl_path.append([0.0]*5)
            else:
                planned_sl_path.append(
                    data["planned_sl_path"]["coeff"] + [data["planned_sl_path"]["max_s"]])
        return obs, ref_line, planned_sl_path

    def calculate_obstacles_mean_std(self):
        all_obstacles=[]
        for obstacles in self.obstacles:
            all_obstacles.extend(obstacles)
        return np.mean(all_obstacles,0),np.std(all_obstacles,0)

    def __getitem__(self, item):
        obstacle = self.obstacles[item]
        normed_obstacle = (obstacle - self.obstacles_mean) / self.obstacles_std

        ref_line = self.ref_line[item]
        normed_ref_line = (ref_line - self.ref_line_mean) / self.ref_line_std

        planned_sl_path = self.planned_sl_path[item]
        normed_planned_sl_path = (
            planned_sl_path - self.planned_sl_path_mean) / self.planned_sl_path_std

        return normed_obstacle, normed_ref_line, normed_planned_sl_path

    def __len__(self):
        return len(self.data_info)


if __name__ == "__main__":
    file_path = './data/pnc_path/cubic_poly/train/0__10_train.json'
    path_dataset = PathDataset(file_path)
    normed_obstacle, normed_ref_line, normed_planned_sl_path = path_dataset[0]
    print(normed_obstacle, normed_ref_line, normed_planned_sl_path)
