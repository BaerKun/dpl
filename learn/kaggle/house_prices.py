import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    # 读取
    dataset_folder = "../../data/house-prices"
    train_pd = pd.read_csv(os.path.join(dataset_folder, "train.csv"))
    test_pd = pd.read_csv(os.path.join(dataset_folder, "test.csv"))

    # 去除ID和SalePrice，保留决定性特征
    all_features = pd.concat((train_pd.drop(columns=["Id", "SalePrice"]), test_pd.drop(columns=["Id"])))

    # 数值数据标准化，NAN置0
    numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda _x: (_x - _x.mean()) / (_x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 对分类数据进行one-hot编码，NAN也视作一个类型
    all_features = pd.get_dummies(all_features, dummy_na=True)

    # 准换为tensor
    num_train = train_pd.shape[0]
    data_np = all_features.values.astype(np.float32)
    label_np = train_pd["SalePrice"].values.reshape(-1, 1).astype(np.float32)

    train_data = torch.tensor(data_np[:num_train], dtype=torch.float32)
    train_label = torch.tensor(label_np, dtype=torch.float32)
    test_data = torch.tensor(data_np[num_train:], dtype=torch.float32)

    return TensorDataset(train_data, train_label), TensorDataset(test_data), data_np.shape[1]


train_dataset, test_dataset, num_features = load_data()
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = torch.nn.Sequential(
    torch.nn.Linear(num_features, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
).to(device)

loss_f = torch.nn.MSELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

for epoch in range(100):
    total_loss = 0.
    for x, l in train_loader:
        x = x.to(device)
        l = l.to(device)
        optimizer.zero_grad()
        y = model(x)
        loss = loss_f(y, torch.log(l))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch: {epoch}, loss: {total_loss}")

predict = torch.zeros((len(test_dataset), 1), dtype=torch.float32, device=device)

with torch.no_grad():
    beginning = 0
    for x, in test_loader:
        x = x.to(device)
        y = model(x)
        torch.exp(y, out=predict[beginning:beginning + len(x)])
        beginning += 128

# export as csv
submission = pd.DataFrame(predict.cpu().detach().numpy(), columns=["SalePrice"])
submission.insert(0, "Id", np.arange(len(train_dataset) + 1, len(train_dataset) + len(test_dataset) + 1))
submission.to_csv("results/submission.csv", index=False)
