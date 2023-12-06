import numpy as np
import random
import matplotlib.pyplot as plt
import math

world_ref_range = (-100., 100.)
max_length = 100.
a0_range = (-100., 100.)
a1_range = (-1., 1.)
a2_range = (-0.01, 0.01)
a3_range = (-0.0001, 0.0001)


def generate_cubic_poly_wrt_x():
    a0 = random.uniform(a0_range[0], a0_range[1])
    a1 = random.uniform(a1_range[0], a1_range[1])
    a2 = random.uniform(a2_range[0], a2_range[1])
    a3 = random.uniform(a3_range[0], a3_range[1])
    poly = np.poly1d([a3, a2, a1, a0])
    return poly


def solve_cubic_poly(x0, y0, dy0, x1, y1, dy1):
    A = np.array([[1., x0, x0 ** 2, x0 ** 3],
                  [1., x1, x1 ** 2, x1 ** 3],
                  [0., 1., 2. * x0, 3. * x0 ** 2],
                  [0., 1., 2. * x1, 3. * x1 ** 2]])
    b = np.array([y0, y1, dy0, dy1])
    return np.linalg.solve(A, b)


def generate_sl_line(s0, s1, l0=0.0, l1=0.0, dl0=0.0, dl1=0.0):
    param = solve_cubic_poly(s0, l0, dl0, s1, l1, dl1)
    return np.poly1d(param[::-1])


class DiscretePath:
    __Max_Offset = 20.0

    def __init__(self, x, y):
        assert isinstance(x, list) and isinstance(y, list)
        assert len(x) == len(y)
        self.path_points = []
        self.range = ((min(x), max(x)), (min(y), max(y)))

        length = 0.0
        for i in range(len(x)):
            delta = 0.0 if i == 0 else math.sqrt(pow(x[i] - x[i - 1], 2) + pow(y[i] - y[i - 1], 2))
            length += delta
            heading = math.atan2(y[i + 1] - y[i], x[i + 1] - x[i]) if i == 0 else math.atan2(y[i] - y[i - 1],
                                                                                             x[i] - x[i - 1])
            self.path_points.append((length, (x[i], y[i], heading)))

    def max_length(self):
        return self.path_points[-1][0]

    def get_points(self, max_length, num=100):
        ret = []
        for s in np.linspace(0.0, max_length, num):
            if self.path_points[0][0] <= s <= self.path_points[-1][0]:
                x, y = self.get_xy(s, 0.0)
                ret.append([x, y])
            else:
                ret.append([0., 0.])
        return ret

    def binary_find_lower_index(self, s):
        min_index = 0
        max_index = len(self.path_points) - 1
        while min_index < max_index - 1:
            middel_index = (min_index + max_index) // 2
            if self.path_points[middel_index][0] > s:
                max_index = middel_index
            elif self.path_points[middel_index][0] < s:
                min_index = middel_index
            else:
                return middel_index
        return min_index

    def get_xy(self, s, l):
        # assert self.path_points[0][0] <= s <= self.path_points[-1][0]
        # assert -self.__Max_Offset <= l <= self.__Max_Offset, f"l:{l} should be in range ({-self.__Max_Offset},{self.__Max_Offset})"
        lower_index = self.binary_find_lower_index(s)
        near_s, near_point = self.path_points[lower_index]
        near_x, near_y, near_heading = near_point
        ref_point_x = near_x + (s - near_s) * math.cos(near_heading)
        ref_point_y = near_y + (s - near_s) * math.sin(near_heading)
        delta_x = l * math.cos(near_heading + math.pi / 2)
        delta_y = l * math.sin(near_heading + math.pi / 2)
        x, y = ref_point_x + delta_x, ref_point_y + delta_y
        return (x, y)

    @staticmethod
    def calculate_distance(point1, point2):
        return math.sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2))

    @staticmethod
    def is_in_box(point, box):
        c_x, c_y, w, h, heading = box
        line1 = (math.cos(heading) * w / 2, math.sin(heading) * w / 2)
        line2 = (math.cos(heading + math.pi / 2) * h / 2, math.sin(heading + math.pi / 2) * h / 2)
        p1 = (c_x + line1[0] + line2[0], c_y + line1[1] + line2[1])
        p2 = (c_x - line1[0] + line2[0], c_y - line1[1] + line2[1])
        p3 = (c_x - line1[0] - line2[0], c_y - line1[1] - line2[1])
        p4 = (c_x + line1[0] - line2[0], c_y + line1[1] - line2[1])
        rec_points = [p1, p2, p3, p4, p1]
        for i in range(len(rec_points) - 1):
            a = (rec_points[i + 1][0] - rec_points[i][0], rec_points[i + 1][1] - rec_points[i][1])
            b = (point[0] - rec_points[i + 1][0], point[1] - rec_points[i + 1][1])
            cross_prod = a[0] * b[1] - b[0] * a[1]
            if cross_prod < 0.:
                return False
        return True

    def is_collision_with_obs(self, obs_box):
        c_x, c_y, w, h, heading = obs_box
        cur_s = 0.0
        while cur_s < self.max_length():
            cur_point = self.get_xy(cur_s, 0.0)
            if self.is_in_box(cur_point, obs_box):
                return True
            else:
                dist = self.calculate_distance(cur_point, (c_x, c_y))
                cur_s += max(dist - max(w, h), 0.25)
        return False

    def show(self, ax: plt.Axes, max_s=100.0,color="b"):
        x = []
        y = []
        for i in range(len(self.path_points)):
            if self.path_points[i][0] > max_s:
                break
            x.append(self.path_points[i][1][0])
            y.append(self.path_points[i][1][1])
        ax.plot(x, y,color=color)
        # plt.show()


def convert_path(poly: np.poly1d, max_x: float, min_x=0.0, step_x=0.25):
    x = []
    y = []
    for s in np.arange(min_x, max_x + step_x, step_x):
        x.append(s)
        y.append(poly(s))
    path = DiscretePath(x, y)
    return path


def get_discrete_point_by_sl_poly(sl_poly: np.poly1d, ref_path: DiscretePath, max_s: float, min_s=0.0, step_s=1.0):
    ret = []
    for s in np.arange(min_s, max_s, step_s):
        ret.append(ref_path.get_xy(s, sl_poly(s)))
    return ret


obs_max_num = 30
obs_l_range = (-20, 20)
obs_heigth_range = (1.5, 6.0)
obs_width_range = (1.5, 6.0)


def generate_obs_along_path(path: DiscretePath, max_s=100.0):
    num_obs = random.randint(0, obs_max_num)
    ret = []
    for i in range(num_obs):
        s = random.uniform(0.0, max_s)
        l = random.uniform(*obs_l_range)
        c_x, c_y = path.get_xy(s, l)
        w = random.uniform(*obs_width_range)
        h = random.uniform(*obs_heigth_range)
        heading = random.uniform(-math.pi, math.pi)
        ret.append((c_x, c_y, w, h, heading))
    return ret


def plot_obs(ax: plt.Axes, obs_list: list):
    for obs in obs_list:
        c_x, c_y, w, h, heading = obs
        rec = plt.Rectangle((c_x - w / 2, c_y - h / 2), w, h, heading * 180 / math.pi, rotation_point='center')
        ax.add_patch(rec)


if __name__ == "__main__":
    while True:
        fig, ax = plt.subplots()
        p = generate_cubic_poly_wrt_x()

        path = convert_path(p, 100.0)
        print(path.max_length())
        path.show(ax)

        x0, y0, dy0, x1, y1, dy1 = 0.0, 0.0, 0.0, 100.0, 5.0, 0.0
        param = solve_cubic_poly(x0, y0, dy0, x1, y1, dy1)
        poly = np.poly1d(param[::-1])
        discrete_point = get_discrete_point_by_sl_poly(poly, path, max_s=x1, step_s=0.1)
        x, y = [], []
        for point in discrete_point:
            x.append(point[0])
            y.append(point[1])
        path2 = DiscretePath(x, y)
        path2.show(ax)

        obs_list = generate_obs_along_path(path)
        plot_obs(ax, obs_list)

        point = (5.5, 2.5)
        box = (50., 0., 16.0, 86.0, 0.5)
        rec = plt.Rectangle((box[0] - box[2] / 2, box[1] - box[3] / 2), box[2], box[3], box[4] * 180 / math.pi,
                            color="b", rotation_point='center')
        ax.add_patch(rec)
        ax.plot(point[0], point[1], "ro")
        print(DiscretePath.is_in_box(point, box))
        print("path.is_collision_with_obs(box):", path.is_collision_with_obs(box))
        plt.show()
        print("path.is_collision_with_obs(box):", path.is_collision_with_obs(box))
