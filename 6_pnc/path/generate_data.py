import copy
import os.path
import random
import json
import numpy as np
from tqdm import tqdm
import threading
from generate_world import *

lock = threading.Lock()


def worker(dataset: list, loop: int):
    for _ in tqdm(range(loop)):
        # mock reference line
        ref_line_max_x = 100.0
        max_length = 100.0
        sample_num = 21
        centric_line_cubic_poly = generate_cubic_poly_wrt_x()
        centric_line_coef = centric_line_cubic_poly.coef

        discrete_centric_line: DiscretePath = convert_path(centric_line_cubic_poly, ref_line_max_x)

        centric_line_discrete_points = discrete_centric_line.get_points(max_length, sample_num)

        # mock obstacle
        obs_list = generate_obs_along_path(discrete_centric_line, max_length)

        # occ_flag = discrete_centric_line.get_occ_status(obs_list, max_length, sample_num)

        # sample path
        max_l = 25.0
        # mock planning start point
        # s0, l0, dl0 = 0.0, 0.0, 0.0
        s0, l0, dl0 = random.uniform(0.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-0.1, 0.1)

        # sample_s = np.arange(max_length, s0, -2.5)
        sample_s = [max_length]
        sample_left_l = [l for l in np.linspace(0.0, max_l, 101)]
        sample_l = [sample_left_l[0]]
        sample_left_l.remove(sample_left_l[0])
        for l in sample_left_l:
            sample_l.extend([-l, l])

        valid_cubic_path = None
        valid_discrete_path = None
        valid_max_s = None
        for s in sample_s:
            for l in sample_l:
                sl_cubic_path = generate_sl_line(s0, s, l0, l, dl0=dl0)
                sample_path_points = get_discrete_point_by_sl_poly(sl_cubic_path, discrete_centric_line, s, s0,
                                                                   step_s=0.25)
                if len(sample_path_points) < 20:
                    continue
                x, y = [], []
                for point in sample_path_points:
                    x.append(point[0])
                    y.append(point[1])
                sample_path = DiscretePath(x, y)
                is_collision_path = False
                for obs in obs_list:
                    is_collision_path = sample_path.is_collision_with_obs(obs)
                    if is_collision_path:
                        break
                if is_collision_path:
                    continue
                else:
                    valid_cubic_path = sl_cubic_path
                    valid_discrete_path = sample_path
                    valid_max_s = s
                    break

            if valid_max_s:
                break

        # for i, f in enumerate(occ_flag):
        #     if f == True:
        #         print([i, f], end=" ")

        # fig, ax = plt.subplots()
        # discrete_centric_line.show(ax)
        # if valid_discrete_path:
        #     valid_discrete_path.show(ax, max_s=valid_discrete_path.max_length(), color='r')
        # plot_obs(ax, obs_list)
        # print("valid_max_s: ", valid_max_s)
        # plt.show()

        # if valid_cubic_path:
        #     print(valid_cubic_path.coef)
        # else:
        #     print("no valid path!")

        planned_discrete_points = valid_discrete_path.get_points(max_length, sample_num) \
            if valid_max_s else [list(discrete_centric_line.get_xy(s0, l0))] * sample_num

        planned_discrete_sl_points = []
        # if valid_max_s:
        #     for s in np.linspace(0.0, max_length, sample_num):
        #         if s > valid_max_s:
        #             planned_discrete_sl_points.append(copy.deepcopy(planned_discrete_sl_points[-1]))
        #         else:
        #             l = valid_cubic_path(s)
        #             dl = valid_cubic_path.deriv()(s)
        #             planned_discrete_sl_points.append([s, l, dl])
        # else:
        #     planned_discrete_sl_points = [[0.0, 0.0, 0.0]] * sample_num

        train_data = {}
        train_data["centric_cubic_poly"] = {"coeff": list(centric_line_coef), "centric_line_max_x": ref_line_max_x}
        train_data['centric_line_discrete_point'] = centric_line_discrete_points
        train_data["obstacles"] = obs_list
        train_data["planned_sl_path"] = {"coeff": list(valid_cubic_path.coef) if valid_max_s else None,
                                         "max_s": valid_max_s if valid_max_s else None}
        train_data['planned_discrete_points'] = planned_discrete_points
        train_data['planned_discrete_sl_points'] = planned_discrete_sl_points
        # train_data['occ_flag'] = occ_flag

        # with lock:
        dataset.append(train_data)


DATASET_SIZE = {"train": 20000, "test": 3000}
batch_size = 4000
for dataset_name, dataset_size in DATASET_SIZE.items():
    batch_num = dataset_size // batch_size
    for i in range(batch_num):
        dataset = []
        worker(dataset, batch_size)
        file_path = f"./data/pnc_path/cubic_poly/{dataset_name}/{i * batch_size}__{(i + 1) * batch_size}_{dataset_name}.json"
        file_dir = os.path.dirname(file_path)
        print(file_dir)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        with open(file_path, "w+") as fb:
            json.dump(dataset, fb)
