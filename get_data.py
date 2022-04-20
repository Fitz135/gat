from collections import OrderedDict
import re
import numpy as np
import math


class DistanceMatrix:
    def __init__(self, path):
        self.path = path + '/'
        self.grid_file = self.path + 'GridList.txt'
        self.cell_file = self.path + 'CellList.txt'
        self.label_file = self.path + 'result'

        self.cell_list = OrderedDict()
        self.grid_list = OrderedDict()

        self.grid_list_info = dict()
        self.cell_list_info = dict()
        self.label_info = []

        self.measure_dist()

    def measure_dist(self):
        self.get_grid_list(self.grid_file)
        self.get_cell_list(self.cell_file)
        self.get_labels(self.label_file)

    def get_grid_list(self, filename):
        with open(filename, 'r') as fp:
            for index, line in enumerate(fp):
                A = re.findall('\d+\.?\d*\.?\d*', line)
                if A[0] not in self.grid_list.keys():
                    self.grid_list[A[0]] = np.array([float(A[1]), float(A[2]), float(A[3]), index])
                    self.grid_list_info[A[0]] = line

    def get_cell_list(self, filename):
        with open(filename, 'r') as fp:
            for index, line in enumerate(fp):
                A = re.findall('\d+.\d+', line)
                if A[0] not in self.cell_list.keys():
                    self.cell_list[A[0]] = np.array([float(A[1]), float(A[2]), float(A[3]), index])
                    self.cell_list_info[A[0]] = line

    def get_labels(self, filenname):
        with open(filenname, 'r') as fp:
            for index, line in enumerate(fp):
                info = []
                info.extend([float(i) for i in line.split()])
                self.label_info.append(info)


