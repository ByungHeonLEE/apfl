import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, subgraph
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
global_model = GCN(in_channels, out_channels)

# 데이터 로더 생성
train_loader = DataLoader([pyg_data], batch_size=1, shuffle=True)
test_loader = DataLoader([pyg_data], batch_size=1, shuffle=False)

# 학습 설정
criterion = torch.nn.CrossEntropyLoss()

# 연합 학습 루프
def federated_train(agent_loaders, global_model, epochs=30):
    global_weights = global_model.state_dict()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01)
    
    epoch_losses = []

    for epoch in range(epochs):
        local_weights = []
        local_losses = []

        for loader in agent_loaders:
            model = GCN(in_channels, out_channels)
            model.load_state_dict(global_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            model.train()
            agent_losses = []
            for data in loader:
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                agent_losses.append(loss.item())

            local_weights.append(model.state_dict())
            local_losses.append(sum(agent_losses) / len(agent_losses))
            print(f'Agent Loss: {sum(agent_losses) / len(agent_losses):.4f}')

        # 글로벌 모델 업데이트 (평균)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        epoch_loss = sum(local_losses) / len(local_losses)
        epoch_losses.append(epoch_loss)
        print(f'Epoch: {epoch+1}, Global Loss: {epoch_loss:.4f}')

    return epoch_losses

def average_weights(weights):
    avg_weights = weights[0].copy()
    for key in avg_weights.keys():
        for i in range(1, len(weights)):
            avg_weights[key] += weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(weights))
    return avg_weights

# 평가 루프
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        out = model(data)
        pred = out[data.test_mask].argmax(dim=1)
        correct += (pred == data.y[data.test_mask]).sum().item()
        total += len(data.test_mask)
    return correct / total

# 연합 학습
num_agents = 3
node_indices = list(range(pyg_data.num_nodes))
split_indices = [node_indices[i::num_agents] for i in range(num_agents)]
agent_data = []
for indices in split_indices:
    mask = torch.tensor(indices, dtype=torch.long)
    subgraph_data = subgraph(mask, pyg_data.edge_index, relabel_nodes=True, num_nodes=pyg_data.num_nodes)
    sub_data = Data(
        x=pyg_data.x[mask],
        edge_index=subgraph_data[0],
        y=pyg_data.y[mask],
        train_mask=torch.tensor([i in train_mask for i in indices], dtype=torch.bool),
        test_mask=torch.tensor([i in test_mask for i in indices], dtype=torch.bool)
    )
    agent_data.append(sub_data)

agent_loaders = [DataLoader([data], batch_size=1, shuffle=True) for data in agent_data]

# 연합 학습 및 성능 평가
epoch_losses = federated_train(agent_loaders, global_model, epochs=30)

# 학습 곡선 시각화 및 저장
def plot_and_save_loss(epoch_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

plot_and_save_loss(epoch_losses)

# 전체 데이터로 평가
accuracy = test(global_model, test_loader)
print(f'Global Model Accuracy: {accuracy:.4f}')
