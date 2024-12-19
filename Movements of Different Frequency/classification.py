import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Data Processing
class DataProcessor:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.rotation_columns = [
            ['Rz', 'Ry', 'Rx'],       # Port 1
            ['Rz.1', 'Ry.1', 'Rx.1'], # Port 2
            ['Rz.2', 'Ry.2', 'Rx.2'], # Port 3
            ['Rz.3', 'Ry.3', 'Rx.3']  # Port 4
        ]

    def load_data(self, class_num, sample_num):
        """
        Load data for a specific class and sample number.

        Args:
            class_num (int): Class number.
            sample_num (int): Sample number.

        Returns:
            pd.DataFrame: DataFrame containing rotation data and labels.
        """
        class_folder = os.path.join(self.data_folder, f'{class_num}_{sample_num:03d}.csv')
        class_data = pd.read_csv(class_folder)
        rotation_data = class_data[[col for group in self.rotation_columns for col in group]]
        labels = np.full(len(rotation_data), class_num)
        class_data['label'] = labels
        return class_data
    
# Define the custom dataset class
class IMUDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (DataFrame): A DataFrame containing rotation data and labels.
        """
        self.data = data.iloc[:, :-1].values  # All columns except the label
        self.labels = data['label'].values    # Labels column

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)




# Visualize a batch of data
def visualize_data(inputs, labels, sample_count=5):
    """
    Visualizes rotation data for a few samples from the batch.

    Args:
        inputs (torch.Tensor): Input data of shape [batch_size, num_features].
        labels (torch.Tensor): Corresponding labels.
        sample_count (int): Number of samples to visualize.
    """
    inputs = inputs.numpy()  # Convert to NumPy for plotting
    labels = labels.numpy()  # Convert to NumPy for labels

    # Plot Rz, Ry, Rx for selected samples
    for i in range(min(sample_count, len(inputs))):
        plt.figure(figsize=(10, 4))
        plt.plot(inputs[i, :3], label='Port 1 (Rz, Ry, Rx)', marker='o')
        plt.plot(inputs[i, 3:6], label='Port 2 (Rz, Ry, Rx)', marker='o')
        plt.plot(inputs[i, 6:9], label='Port 3 (Rz, Ry, Rx)', marker='o')
        plt.plot(inputs[i, 9:12], label='Port 4 (Rz, Ry, Rx)', marker='o')
        plt.title(f"Sample {i+1} (Label: {int(labels[i])})")
        plt.xlabel("Axes (Rz, Ry, Rx)")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        plt.show()




if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    # Adding paths (if required)
    data_folder_1 = r'C:\Users\11424\Documents\phd\Movements of Different Frequency\pwm_37'
    data_folder_2 = r'C:\Users\11424\Documents\phd\Movements of Different Frequency\pwm_40'

    # Define the dataset class
    class1 = pd.read_csv(data_folder_1 + '/1_000.csv')
    class2 = pd.read_csv(data_folder_2 + '/1_000.csv')

    # Display the first few rows of the dataset
    # print(class1.head())
    rotation_x_columns = [['Rx', 'Rx.1', 'Rx.2', 'Rx.3']]
    rotation_y_columns = [['Ry', 'Ry.1', 'Ry.2', 'Ry.3']]
    rotation_z_columns = [['Rz', 'Rz.1', 'Rz.2', 'Rz.3']]

    [
        ['Rz', 'Ry', 'Rx'],       # Port 1
        ['Rz.1', 'Ry.1', 'Rx.1'], # Port 2
        ['Rz.2', 'Ry.2', 'Rx.2'], # Port 3
        ['Rz.3', 'Ry.3', 'Rx.3']  # Port 4
    ]

    rotation_x_class_1 = class1[[col for group in  rotation_x_columns for col in group]]
    rotation_y_class_1 = class1[[col for group in  rotation_y_columns for col in group]]
    rotation_z_class_1 = class1[[col for group in  rotation_z_columns for col in group]]
    print(rotation_x_class_1.shape)
    print(rotation_x_class_1.head())
    # # cut the data to 50 samples


    # data_1 = rotation_data_class_1.copy()
    # labels_class_1 = np.zeros(len(data_1))
    # data_1['label']= labels_class_1
    # rotation_data_class_2 = class2[[col for group in rotation_columns for col in group]]
    # data_2 = rotation_data_class_2.copy()
    # labels_class_2 = np.ones(len(data_2))
    # data_2['label']= labels_class_2

    # # print(rotation_data_class_1.head())
    # # print(rotation_data_class_2.head())
    # # Combine datasets
    # combined_data = pd.concat([data_1, data_2], ignore_index=True)

    # # Create dataset and DataLoader
    # imu_dataset = IMUDataset(combined_data)
    # data_loader = DataLoader(imu_dataset, batch_size=64, shuffle=True, num_workers=4)

    # # # Example usage
    # # for inputs, labels in data_loader:
    # #     print("Batch Inputs Shape:", inputs.shape)
    # #     print("Batch Labels Shape:", labels.shape)
    # #     break

    # for inputs, labels in data_loader:
    #     visualize_data(inputs, labels, sample_count=5)
    #     break

