from torch_geometric.data import DataLoader

# 데이터 로더 생성
loader = DataLoader([pyg_data], batch_size=1, shuffle=True)

# 학습 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 학습 루프
def train():
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
    return loss.item()

# 평가 루프
def test():
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# 학습 및 평가
for epoch in range(1, 201):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
