from model_architecture import NeuralNetwork as NN
import torch
import pandas as pd
import joblib as jl
import torch.nn as nn
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS


dataset = pd.read_csv("dataset.csv")

x = dataset.drop("Wine", axis=1)
y = dataset["Wine"]


x_train, x_test, y_train, y_test = tts(x,y, test_size = 0.20, train_size = 0.80, random_state=42, shuffle=True)

scaler = SS()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

jl.dump(scaler, "scaler.pkl")

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()

y_train = torch.tensor(y_train.to_numpy()).long()
y_test = torch.tensor(y_test.to_numpy()).long()

instance = NN(input_size=13, hidden_size=16, num_classes=3) # we use 13, 16, 3


#training loop

criteria = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(instance.parameters(), lr=lr)


for epoch in range(1000):
    output = instance(x_train)
    loss = criteria(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    

instance.eval()

with torch.no_grad():
    output = instance(x_test)
    pred = torch.argmax(output, dim=1)
    
    correct = (pred == y_test).sum()
    total = len(y_test)
    
    accuracy = correct/total*100
    print("accuracy = ", accuracy.item(),"%")
    
with torch.inference_mode():
    train_output = instance(x_train)
    train_pred = torch.argmax(train_output, dim=1)

    train_accuracy = (train_pred == y_train).float().mean()

print(f"Train Accuracy: {train_accuracy.item() * 100:.2f}%")
    

torch.save(instance.state_dict(), "W.I.N.E..pth")
