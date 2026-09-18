from model_architecture import NeuralNetwork
import torch
import pandas as pd
import joblib

model = NeuralNetwork(13,16,3)


model.load_state_dict(torch.load("W.I.N.E..pth"))
model.eval()

test_data = pd.read_csv("test_data.csv")

x_test = test_data.drop("Wine", axis=1)
y_test = test_data["Wine"]

scaler = joblib.load("scaler.pkl")

x_test = scaler.transform(x_test)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy()).long()

with torch.inference_mode():
    outputs = model(x_test)

prediction = torch.argmax(outputs[0])

if prediction == 1:
    print("Wine class detected! Classified as class 1")
elif prediction == 2:
    print("Wine class detected! Classified as class 2")
elif prediction == 0:
    print("Wine class detected! Classified as class 0")
else:
    print("ERROR")
    

