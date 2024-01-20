import pickle

# Replace this with the path to your .pkl file
file_path = r"E:\Datasets\IEMOCAP_features.pkl"

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Use pickle.load to deserialize the data
    iemocap_features = pickle.load(file)

# iemocap_features now contains the deserialized Python object
print(type(iemocap_features))  # Check the type of the loaded object

# If it's a dictionary, you can check its keys
if isinstance(iemocap_features, dict):
    print(iemocap_features.keys())
# Check the length of the list
print("Number of elements in the dataset:", len(iemocap_features))

# Inspect the first few elements
for i in range(5):
    print(f"Element {i}: {iemocap_features[i]}")
    # Add more specific inspections here depending on your data structure
