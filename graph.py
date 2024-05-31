import networkx as nx
import pandas as pd
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt

# CSV 파일 읽기
file_path = 'combined_dataset.csv'  # 통합된 CSV 파일 경로
df = pd.read_csv(file_path)

def dataframe_to_graph(df):
    G = nx.DiGraph()
    
    for index, row in df.iterrows():
        G.add_edge(row['From'], row['To'], TxHash=row['TxHash'], BlockHeight=row['BlockHeight'], 
                TimeStamp=row['TimeStamp'], Value=row['Value'], isError=row['isError'])
    return G

def graph_to_pyg_data(graph):
    return from_networkx(graph)

graph = dataframe_to_graph(df)
print(graph.graph)


# 그래프 시각화
def visualize_graph(graph, num_nodes=50):
    plt.figure(figsize=(12, 8))
    subgraph = graph.subgraph(list(graph.nodes)[:num_nodes])  # 일부 노드만 시각화
    pos = nx.spring_layout(subgraph)  # 레이아웃 설정
    nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()

visualize_graph(graph)

pyg_data = graph_to_pyg_data(graph)
print(pyg_data)

# PyTorch Geometric 데이터 속성 확인
print("PyTorch Geometric Data:")
print(f"Edge Index: {pyg_data.edge_index}")
print(f"TxHash: {pyg_data.TxHash}")
print(f"BlockHeight: {pyg_data.BlockHeight}")
print(f"TimeStamp: {pyg_data.TimeStamp}")
print(f"Value: {pyg_data.Value}")
print(f"isError: {pyg_data.isError}")
print(f"Number of nodes: {pyg_data.num_nodes}")

