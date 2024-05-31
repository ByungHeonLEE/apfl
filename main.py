import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # sklearn에서 train_test_split 가져오기


# CSV 파일 읽기
file_path = 'combined_dataset.csv'  # 통합된 CSV 파일 경로
df = pd.read_csv(file_path)

# 데이터프레임을 NetworkX 그래프로 변환하는 함수
def dataframe_to_graph(df):
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['From'], row['To'], TxHash=row['TxHash'], BlockHeight=row['BlockHeight'], 
                TimeStamp=row['TimeStamp'], Value=row['Value'], isError=row['isError'])
    return G

# NetworkX 그래프를 PyTorch Geometric 데이터로 변환하는 함수
def graph_to_pyg_data(graph):
    pyg_data = from_networkx(graph)

    # 노드 특성을 추가해야 함
    # 여기서는 각 노드에 임의의 특성을 부여 (실제 데이터에 맞게 수정 필요)
    node_features = torch.ones((pyg_data.num_nodes, 2))  # 예시로 모든 노드에 두 개의 특성을 1로 설정
    pyg_data.x = node_features

    # 엣지 레이블 추가 (예시로 각 엣지의 레이블을 0으로 설정, 실제 데이터에 맞게 수정 필요)
    pyg_data.y = torch.zeros(pyg_data.num_nodes, dtype=torch.long)

    return pyg_data

# 그래프로 변환
graph = dataframe_to_graph(df)

# PyTorch Geometric 데이터로 변환
pyg_data = graph_to_pyg_data(graph)

# 데이터를 학습 세트와 테스트 세트로 분리
train_mask, test_mask = train_test_split(range(pyg_data.num_nodes), test_size=0.2, random_state=42)

# 마스크를 사용하여 데이터 분할
pyg_data.train_mask = torch.tensor(train_mask, dtype=torch.long)
pyg_data.test_mask = torch.tensor(test_mask, dtype=torch.long)

# def visualize_graph(graph, num_nodes=50):
#     plt.figure(figsize=(12, 8))
#     subgraph = graph.subgraph(list(graph.nodes)[:num_nodes])  # 일부 노드만 시각화
#     pos = nx.spring_layout(subgraph)  # 레이아웃 설정
#     nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
#     plt.show()

# visualize_graph(graph)

# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 입력 채널과 출력 채널 설정 (입력 채널 수는 노드 특성 수, 출력 채널 수는 클래스 수)
in_channels = pyg_data.num_node_features  # 노드 특성 수
out_channels = 2  # Fraud/Not-Fraud 이진 분류
model = GCN(in_channels, out_channels)

# 데이터 로더 생성
# loader = DataLoader([pyg_data], batch_size=1, shuffle=True)
# 데이터 로더 생성
train_loader = DataLoader([pyg_data], batch_size=1, shuffle=True)
test_loader = DataLoader([pyg_data], batch_size=1, shuffle=False)

# 학습 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 학습 루프
def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return loss.item()

# 평가 루프
def test():
    model.eval()
    correct = 0
    for data in test_loader:
        out = model(data)
        pred = out[data.test_mask].argmax(dim=1)
        correct += (pred == data.y[data.test_mask]).sum().item()
    return correct / len(data.test_mask)

# 학습 및 평가
for epoch in range(1, 201):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
