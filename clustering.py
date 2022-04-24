import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import copy
import pandas as pd
import time
import torch
import random
from get_data import DistanceMatrix

from scipy import spatial

EPSILON_PRIMAL = 1e-5
EPSILON_DUAL = 1e-5


class Clustering:
    grid_list = OrderedDict()
    cell_list = OrderedDict()
    grid_list_info = dict()
    cell_list_info = dict()
    adj = []

    def __init__(self, path):
        dis_matrix = DistanceMatrix(path)
        self.grid_list = dis_matrix.grid_list
        self.cell_list = dis_matrix.cell_list
        self.grid_list_info = dis_matrix.grid_list_info
        self.cell_list_info = dis_matrix.cell_list_info
        self.labels_info = dis_matrix.label_info

    def miller_to_xy(self, lon, lat):
        xy_coordinate = []
        l = 6381372 * math.pi * 2
        w = l
        h = l / 2
        mill = 2.3
        x = lon * math.pi / 180
        y = lat * math.pi / 180
        y = 1.25
        y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))
        x = (w / 2) + (w / (2 * math.pi)) * x
        y = (h / 2) - (h / (2 * mill)) * y
        xy_coordinate.append((int(round(x))))
        xy_coordinate.append((int(round(y))))

        return xy_coordinate

    def cluster(self):
        grid_position = []
        lonlat_grid = np.array(list(self.grid_list.values()))[:, 0:3]
        for i, value in enumerate(lonlat_grid):
            id = i
            la = value[0]
            lo = value[2]
            info = [id, lo, la]
            grid_position.append(info)

        cell_grid = dict()

        cell_position = []
        x = []
        lonlat_cell = np.array(list(self.cell_list.values()))[:, 0:2]
        for i, value in enumerate(lonlat_cell):
            id = i
            la = value[1]
            lo = value[0]
            info = [id, lo, la]
            pos = [lo, la]
            x.append(pos)
            cell_position.append(info)
            cell_grid[id] = []
        x = torch.tensor(x,dtype=torch.float)

        for i, grid in enumerate(grid_position):
            min_dis1 = 100000000
            min_dis2 = 100000000
            id1 = -1
            id2 = -1
            for j, cell in enumerate(cell_position):
                dis = math.sqrt((cell[1] - grid[1]) ** 2 + (cell[2] - grid[2]) ** 2)
                if dis < min_dis1:
                    min_dis2 = min_dis1
                    id2 = id1
                    min_dis1 = dis
                    id1 = j
                elif dis == min_dis1:
                    if cell_grid[id1] == []:
                        continue
                    else:
                        id1 = j
                elif dis > min_dis1 and dis < min_dis2:
                    min_dis2 = dis
                    id2 = j
                elif dis == min_dis2:
                    if cell_grid[id2] == []:
                        continue
                    else:
                        id2 = j
            cell_grid[id1].append(i)
            cell_grid[id2].append(i)

        edge_index = []
        first_e = []
        second_e = []
        edge_index.append(first_e)
        edge_index.append(second_e)
        l = len(cell_position)
        self.adj = np.zeros((l, l), dtype=int)
        for i in range(l):
            for j in range(l):
                if i == j:
                    continue
                else:
                    res = [v for v in cell_grid[i] if v in cell_grid[j]]
                    if res != []:
                        first_e.append(i)
                        # first_e.append(j)
                        second_e.append(j)
                        # second_e.append(i)

        edge_index = torch.tensor(edge_index,dtype=torch.long)

        y = self.labels_info
        y = torch.tensor(y, dtype=float)

        # train_mask, val_mask, test_mask = [], [], []
        #
        # for i in range(28):
        #     r = random.randint(1,3)
        #     if r == 1:
        #         train_mask.append(True)
        #         val_mask.append(False)
        #         test_mask.append(False)
        #     elif r == 2:
        #         train_mask.append(False)
        #         val_mask.append(True)
        #         test_mask.append(False)
        #     else:
        #         train_mask.append(False)
        #         val_mask.append(False)
        #         test_mask.append(True)
        #
        # train_mask = torch.tensor(train_mask)
        # test_mask = torch.tensor(test_mask)
        # val_mask = torch.tensor(val_mask)

        train_mask = torch.tensor([True, False, True,  True, False, False,  True, False, True, False,
        False, False, False, False, False, False, False,  True, False, False,
         True, False,  True, True, False, False, False, False])

        val_mask = torch.tensor([False, False,  False, False,  True, False, False,  True,  False,  True,
        False,  True, False,  True, False,  True,  True, False, False, False,
        False,  True, False, False,  True,  True, False, False])

        test_mask = torch.tensor([False,  True, False, False, False,  True, False, False, False, False,
         True, False,  True, False,  True, False, False, False,  True,  True,
        False, False, False,  False, False, False,  True,  True])

        return x, edge_index, y, train_mask, val_mask, test_mask


