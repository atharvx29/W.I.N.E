from model import NeuralNetwork as NN
import torch
import pandas as pd
import joblib as jl
import torch.nn as nn
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS


dataset = pd.read_csv("dataset.csv")

x = dataset.drop("dataset.csv", 1)
y = dataset["wine"]

y = y-1

x_train, x_test, y_train, y_test = tts(x,y, test_size = 0.20, train_size = 0.80, random_state=42, shuffle=True)

scaler = SS()

x_train = SS.fit_transform(x_train)
x_test = SS.transform(x_test)

jl.dump(scaler, "test_tensor.pkl")

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()

y_train = torch.from_numpy(y_train.to_numpy()).long()
y_test = torch.from_numpy(y_test.to_numpy()).long()

instance = NN(input_size=13, hidden_size=16, num_classes=3) # we use 13, 16, 3




