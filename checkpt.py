import torch

# Load the file
pt_file = torch.load("./Data_new/CEXP/processed/data.pt")

# Print the head of the file
# print(pt_file[:5])

print(len(pt_file))

print(type(pt_file))


print(pt_file)
for data in pt_file:
    print(data.x)
