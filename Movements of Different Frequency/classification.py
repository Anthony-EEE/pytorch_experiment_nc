import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt

# Data Processing
class DataProcessor:
    def __init__(self):
        self.data = None

    def fill_missing_data(self, df_1, missing_value=-3.697314e+28):
        # Replace missing_value with NaN
        data = df_1.values
        data = np.where(data == missing_value, np.nan, data)
        
        # Create a DataFrame for easier handling
        df = pd.DataFrame(data)
        
        # Apply spline interpolation to each column
        for col in df.columns:
            # Interpolate only if there are NaNs
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method='spline', order=3, limit_direction='both')
        
        return df
    
    def apply_zero_mean(self, df):
        """
        Adjust the data so that the first value in each column is zero.
        Args:
            df (DataFrame): DataFrame containing rotation data.
        Returns:
            DataFrame: Adjusted data.
        """
        data = df.values
        for i in range(data.shape[1]):
            delta = -data[0, i]  # Calculate delta to make the first value zero
            data[:, i] += delta  # Add delta to all rows in the column
        return pd.DataFrame(data, columns=df.columns)
    
    def detect_start_indices(self, transition_data, start_offset=400, threshold=1, min_distance=680):
        diff = np.diff(transition_data[start_offset:])
        indices = np.where(np.abs(diff) > threshold)[0]
        refined_indices = [indices[0]]
        for idx in indices[1:]:
            if idx - refined_indices[-1] >= min_distance:
                refined_indices.append(idx)
        refined_indices = np.array(refined_indices) + start_offset

        def find_max_values_and_indices(transition_data, refined_start_indices, range_length=200):
            max_values = []
            max_indices = []
            for idx in refined_start_indices:
                segment = transition_data[idx:(idx + range_length)]
                max_val = np.max(segment)
                max_idx = np.argmax(segment)
                max_values.append(max_val)
                max_indices.append(max_idx)
            # Convert max_indices to global indices by adding the refined start indices
            adjusted_start_indices = np.array(max_indices) +1 + refined_start_indices
            return max_values, adjusted_start_indices
        max_val, indices = find_max_values_and_indices(transition_data, refined_indices)
        return indices
    
    def construct_dataset(self, start_indices, data_x, data_y, data_z, sequence_length):
        data = []
        for idx in start_indices:
            segment = np.concatenate([
                data_x.iloc[idx:idx + sequence_length].values,
                data_y.iloc[idx:idx + sequence_length].values,
                data_z.iloc[idx:idx + sequence_length].values
            ], axis=1)
            data.append(segment)
        return np.array(data)
# Define the custom dataset class
import torch
from torch.utils.data import Dataset

class RotationDataset(Dataset):
    def __init__(self, class_1, class_2):
        """
        Args:
            class_1 (np.ndarray or torch.Tensor): Dataset for class 1 with shape (50, 100, 12)
            class_2 (np.ndarray or torch.Tensor): Dataset for class 2 with shape (50, 100, 12)
        """
        # Convert to PyTorch tensors if they're NumPy arrays
        if isinstance(class_1, np.ndarray):
            class_1 = torch.tensor(class_1, dtype=torch.float32)
        if isinstance(class_2, np.ndarray):
            class_2 = torch.tensor(class_2, dtype=torch.float32)
        
        # Combine datasets and create labels
        self.data = torch.cat([class_1, class_2], dim=0)  # Combine along the first dimension
        self.labels = torch.cat([torch.zeros(class_1.size(0)), torch.ones(class_2.size(0))])  # 0 for class_1, 1 for class_2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]





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
    t=100
    print("PyTorch Version: ",torch.__version__)
    dp = DataProcessor()
    # Adding paths (if required)
    data_folder_1 = r'C:\Users\11424\Documents\phd\pytorch_experiment_nc\Movements of Different Frequency\pwm_37'
    data_folder_2 = r'C:\Users\11424\Documents\phd\pytorch_experiment_nc\Movements of Different Frequency\pwm_40'

    # Load data
    class1 = pd.read_csv(data_folder_1 + '/1_000.csv')
    class2 = pd.read_csv(data_folder_2 + '/1_000.csv')
    print(class1.head())

    #extracting transition data
    transition_x_columns = [['Tx', 'Tx.1', 'Tx.2', 'Tx.3']]
    transition_y_columns = [['Ty', 'Ty.1', 'Ty.2', 'Ty.3']]
    transition_z_columns = [['Tz', 'Tz.1', 'Tz.2', 'Tz.3']]
    # Fill missing data and extract transition data
    transition_x_class_1 = dp.fill_missing_data(class1[[col for group in transition_x_columns for col in group]])
    transition_y_class_1 = dp.fill_missing_data(class1[[col for group in transition_y_columns for col in group]])
    transition_z_class_1 = dp.fill_missing_data(class1[[col for group in transition_z_columns for col in group]])
    print(transition_x_class_1.shape)
    transition_x_class_2 = dp.fill_missing_data(class2[[col for group in  transition_x_columns for col in group]])
    transition_y_class_2 = dp.fill_missing_data(class2[[col for group in  transition_y_columns for col in group]])
    transition_z_class_2 = dp.fill_missing_data(class2[[col for group in  transition_z_columns for col in group]])
    print(transition_x_class_2.shape)

    # Apply zero mean adjustment
    transition_x_class_1 = dp.apply_zero_mean(transition_x_class_1)
    transition_y_class_1 = dp.apply_zero_mean(transition_y_class_1)
    transition_z_class_1 = dp.apply_zero_mean(transition_z_class_1)

    # Start indices detection

    
    start_indices_class1 = dp.detect_start_indices(transition_x_class_1.iloc[:, 0].values)
    start_indices_class2 = dp.detect_start_indices(transition_x_class_2.iloc[:, 0].values)

    print("Refined Start Indices (Class 1):", start_indices_class1)
    for index in start_indices_class1:
        print(transition_x_class_1.iloc[index, 0])
    print("Refined Start Indices (Class 2):", start_indices_class2)

    
    #extracting rotation data
    rotation_x_columns = [['Rx', 'Rx.1', 'Rx.2', 'Rx.3']]
    rotation_y_columns = [['Ry', 'Ry.1', 'Ry.2', 'Ry.3']]
    rotation_z_columns = [['Rz', 'Rz.1', 'Rz.2', 'Rz.3']]

    # Fill missing data and extract rotation data
    rotation_x_class_1 = dp.fill_missing_data(class1[[col for group in rotation_x_columns for col in group]])
    rotation_y_class_1 = dp.fill_missing_data(class1[[col for group in rotation_y_columns for col in group]])
    rotation_z_class_1 = dp.fill_missing_data(class1[[col for group in rotation_z_columns for col in group]])
    print(rotation_x_class_1.shape)
    rotation_x_class_2 = dp.fill_missing_data(class2[[col for group in  rotation_x_columns for col in group]])
    rotation_y_class_2 = dp.fill_missing_data(class2[[col for group in  rotation_y_columns for col in group]])
    rotation_z_class_2 = dp.fill_missing_data(class2[[col for group in  rotation_z_columns for col in group]])
    print(rotation_x_class_2.shape)


    # Apply zero mean adjustment
    rotation_x_class_1 = dp.apply_zero_mean(rotation_x_class_1)
    rotation_y_class_1 = dp.apply_zero_mean(rotation_y_class_1)
    rotation_z_class_1 = dp.apply_zero_mean(rotation_z_class_1)

    rotation_x_class_2 = dp.apply_zero_mean(rotation_x_class_2)
    rotation_y_class_2 = dp.apply_zero_mean(rotation_y_class_2)
    rotation_z_class_2 = dp.apply_zero_mean(rotation_z_class_2)

    # Construct dataset
    start_indices_class1 = start_indices_class1.tolist()
    start_indices_class2 = start_indices_class2.tolist()
    rotation_class_1 = dp.construct_dataset(start_indices_class1, rotation_x_class_1, rotation_y_class_1, rotation_z_class_1, sequence_length=t)
    print(rotation_class_1.shape)
    rotation_class_2 = dp.construct_dataset(start_indices_class2, rotation_x_class_2, rotation_y_class_2, rotation_z_class_2, sequence_length=t)
    print(rotation_class_2.shape)


    # Create the dataset
    dataset = RotationDataset(rotation_class_1, rotation_class_2)

    # Check dataset size
    print(f'Sample size {len(dataset)}')
    print(f"time steps: {len(dataset[0][0])}")
    print(f"Sample data shape: {dataset[0][0].shape}, Label: {dataset[0][1]}")
    # # cut the rotation data to multiple samples
    # # data_1 = rotation_data_class_1.copy()
    # # labels_class_1 = np.zeros(len(data_1))
    # # data_1['label']= labels_class_1
    # # rotation_data_class_2 = class2[[col for group in rotation_columns for col in group]]
    # # data_2 = rotation_data_class_2.copy()
    # # labels_class_2 = np.ones(len(data_2))
    # # data_2['label']= labels_class_2

    # # # print(rotation_data_class_1.head())
    # # # print(rotation_data_class_2.head())
    # # # Combine datasets
    # # combined_data = pd.concat([data_1, data_2], ignore_index=True)

    # # # Create dataset and DataLoader
    # # imu_dataset = IMUDataset(combined_data)
    # # data_loader = DataLoader(imu_dataset, batch_size=64, shuffle=True, num_workers=4)

    # # # # Example usage
    # # # for inputs, labels in data_loader:
    # # #     print("Batch Inputs Shape:", inputs.shape)
    # # #     print("Batch Labels Shape:", labels.shape)
    # # #     break

    # # for inputs, labels in data_loader:
    # #     visualize_data(inputs, labels, sample_count=5)
    # #     break

