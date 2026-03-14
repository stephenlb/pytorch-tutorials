import torch
import datetime


tensor = torch.ones(2000,2000)

## Warm Up
print("Warming up")
for i in range(1000): _ = tensor @ tensor
for i in range(1000): _ = tensor @ tensor.T

## Transposed
print("Starting Transposed Test")
start = datetime.datetime.now()
for i in range(2000): _ = tensor.T @ tensor
end = datetime.datetime.now()
print(f"Latency: {end - start}")

## NOT Transposed
print("Starting Non-Transposed Test")
start = datetime.datetime.now()
for i in range(2000): _ = tensor @ tensor
end = datetime.datetime.now()
print(f"Latency: {end - start}")
