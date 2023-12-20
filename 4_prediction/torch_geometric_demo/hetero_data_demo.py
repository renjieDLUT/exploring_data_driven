from torch_geometric.data import HeteroData
import torch

'''
异构图数据中,不同类型的节点可以存在一个图中,不同类型的节点具有不同的特征和属性.
HeteroData 异构图数据类型,包含不同类型节点的信息,和不同节点之间的连接关系
利于进行图神经网络的训练和推理
'''

data = HeteroData()

num_paper = 736389
num_feature_paper = 256

num_author = 1134649
num_feature_author = 256

num_institution = 8740
num_feature_institution = 256

num_field = 59965
num_feature_field = 256

num_edges_cites = 5416271
num_edges_writes = 7145660
num_edges_affiliated = 1043998
num_edges_topic = 7505078

num_features_cites = 128
num_features_writes = 128
num_features_affiliated = 128
num_features_topic = 128
data['paper'].x = torch.randn(num_paper, num_feature_paper)
data['author'].x = torch.randn(num_author, num_feature_author)
data['institution'].x = torch.randn(num_institution, num_feature_institution)
data['field'].x = torch.randn(num_field, num_feature_field)

data['paper', 'cites', 'paper'].edge_index = [2, num_edges_cites]
data['author', 'writes', 'paper'].edge_index = [2, num_edges_writes]
data['author', 'affiliated_with', 'institution'].edge_index = [2, num_edges_affiliated]
data['paper', 'has_topic', 'field_of_study'].edge_index = [2, num_edges_topic]

data['paper', 'cites', 'paper'].edge_attr = [num_edges_cites, num_features_cites]
data['author', 'writes', 'paper'].edge_attr = [num_edges_writes, num_features_writes]
data['author', 'affiliated_with', 'institution'].edge_attr = [num_edges_affiliated, num_features_affiliated]
data['paper', 'has_topic', 'field_of_study'].edge_attr = [num_edges_topic, num_features_topic]

from torch_geometric.datasets import OGB_MAG

dataset=OGB_MAG('./data/ogb_mag')
data=dataset[0]
print(data)
paper_node_data=data['paper']
cites_node_data=data['paper','cites','paper']
cites_node_data=data['paper','paper']
cites_node_data=data['cites']

data['paper'].year=[10,50]
print(data)
