import torch
from torch_geometric.data import InMemoryDataset
import pickle
import os
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
NAME = "GRAPHSAT"


def bump(g):
    return Data.from_dict(g.__dict__)

def convert_g(g):
    g = Data(**g.__dict__)
    return g

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = (torch.load(self.processed_paths[0]))

    @property
    def raw_file_names(self):
        return [NAME+".pkl"]


    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/"+NAME+".pkl"), "rb"))

        # print("------- check data process 1 ------------")
        # pre_data = data_list[0]
        # for data in data_list:
        #     if len(pre_data.x) == len(data.x):
        #         print("Same? ", torch.all(torch.eq(pre_data.x, data.x)))
        #     pre_data = data

        # data_list = [convert_g(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # print("------- check data process 2 ------------")
        # pre_data = data_list[0]
        # print(pre_data.x)
        # for data in data_list:
        #     if len(pre_data.x) == len(data.x):
        #         print("Same? ", torch.all(torch.eq(pre_data.x, data.x)))
        #     pre_data = data

        # print("len(data_list)", len(data_list))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    test_path = "Data/EXP/"
    dataset = PlanarSATPairsDataset(test_path)
    print(dataset[0])