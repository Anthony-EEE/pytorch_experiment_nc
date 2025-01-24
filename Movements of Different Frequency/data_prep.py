import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt

# Data Processing
class DataProcessor:
    def __init__(self, data_folder_1, data_folder_2, t):
        self.class1 = pd.read_csv(data_folder_1 + '/1_000.csv')
        self.class2 = pd.read_csv(data_folder_2 + '/1_000.csv')
        self.t  = t

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
    
    # def apply_zero_mean(self, df):
    #     """
    #     Adjust the data so that the first value in each column is zero.
    #     Args:
    #         df (DataFrame): DataFrame containing rotation data.
    #     Returns:
    #         DataFrame: Adjusted data.
    #     """
    #     data = df.values
    #     for i in range(data.shape[1]):
    #         delta = -data[0, i]  # Calculate delta to make the first value zero
    #         data[:, i] += delta  # Add delta to all rows in the column
    #     return pd.DataFrame(data, columns=df.columns)
    def apply_zero_mean_pair(self, df_class_1, df_class_2):
        """
        Adjust the data in two DataFrames so that the first value in each column of df_class_1 becomes zero,
        and the same adjustment is applied to the corresponding column in df_class_2.

        Args:
            df_class_1 (DataFrame): DataFrame containing data for class 1.
            df_class_2 (DataFrame): DataFrame containing data for class 2.

        Returns:
            Tuple[DataFrame, DataFrame]: Adjusted data for both class 1 and class 2.
        """
        data_class_1 = df_class_1.values
        data_class_2 = df_class_2.values
        # res = np.zeros(4)
        for i in range(data_class_1.shape[1]):
            delta = -data_class_1[0, i]  # Calculate delta to make the first value of class_1 zero
            data_class_1[0:, i] += delta  # Adjust class_1 column
            data_class_2[0:, i] += delta  # Adjust class_2 column by the same delta
        # print("delta", data_class_1[0, 0], data_class_1[0, 1], data_class_1[0, 2], data_class_1[0, 3])   
        adjusted_class_1 = pd.DataFrame(data_class_1, columns=df_class_1.columns)
        adjusted_class_2 = pd.DataFrame(data_class_2, columns=df_class_2.columns)
        
        return adjusted_class_1, adjusted_class_2
    
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

    def build(self):
        #extracting transition data
        transition_x_columns = [['Tx', 'Tx.1', 'Tx.2', 'Tx.3']]
        transition_y_columns = [['Ty', 'Ty.1', 'Ty.2', 'Ty.3']]
        transition_z_columns = [['Tz', 'Tz.1', 'Tz.2', 'Tz.3']]
        # Fill missing data and extract transition data
        transition_x_class_1 = self.fill_missing_data(self.class1[[col for group in transition_x_columns for col in group]])
        transition_y_class_1 = self.fill_missing_data(self.class1[[col for group in transition_y_columns for col in group]])
        transition_z_class_1 = self.fill_missing_data(self.class1[[col for group in transition_z_columns for col in group]])
        # print(transition_x_class_1.shape)
        transition_x_class_2 = self.fill_missing_data(self.class2[[col for group in  transition_x_columns for col in group]])
        transition_y_class_2 = self.fill_missing_data(self.class2[[col for group in  transition_y_columns for col in group]])
        transition_z_class_2 = self.fill_missing_data(self.class2[[col for group in  transition_z_columns for col in group]])
        # print(transition_x_class_2.shape)

        start_indices_class1 = self.detect_start_indices(transition_x_class_1.iloc[:, 0].values)
        start_indices_class2 = self.detect_start_indices(transition_x_class_2.iloc[:, 0].values)
        # print("Start Indices (Class 1):", start_indices_class1)
        # print("Start Indices (Class 2):", start_indices_class2)
        # print("Refined Start Indices (Class 1):", start_indices_class1)
        # for index in start_indices_class1:
        #     print(transition_x_class_1.iloc[index, 0])
        # print("Refined Start Indices (Class 2):", start_indices_class2)

        
        #extracting rotation data
        rotation_x_columns = [['Rx', 'Rx.1', 'Rx.2', 'Rx.3']]
        rotation_y_columns = [['Ry', 'Ry.1', 'Ry.2', 'Ry.3']]
        rotation_z_columns = [['Rz', 'Rz.1', 'Rz.2', 'Rz.3']]

        # Fill missing data and extract rotation data
        rotation_x_class_1 = self.fill_missing_data(self.class1[[col for group in rotation_x_columns for col in group]])
        rotation_y_class_1 = self.fill_missing_data(self.class1[[col for group in rotation_y_columns for col in group]])
        rotation_z_class_1 = self.fill_missing_data(self.class1[[col for group in rotation_z_columns for col in group]])
        # print(rotation_x_class_1.shape)
        rotation_x_class_2 = self.fill_missing_data(self.class2[[col for group in  rotation_x_columns for col in group]])
        rotation_y_class_2 = self.fill_missing_data(self.class2[[col for group in  rotation_y_columns for col in group]])
        rotation_z_class_2 = self.fill_missing_data(self.class2[[col for group in  rotation_z_columns for col in group]])
        # print(rotation_x_class_2.shape)


        # Apply zero mean adjustment
        # rotation_x_class_1 = self.apply_zero_mean(rotation_x_class_1)
        # rotation_y_class_1 = self.apply_zero_mean(rotation_y_class_1)
        # rotation_z_class_1 = self.apply_zero_mean(rotation_z_class_1)

        # rotation_x_class_2 = self.apply_zero_mean(rotation_x_class_2)
        # rotation_y_class_2 = self.apply_zero_mean(rotation_y_class_2)
        rotation_x_class_1, rotation_x_class_2 = self.apply_zero_mean_pair(rotation_x_class_1, rotation_x_class_2)
        rotation_y_class_1, rotation_y_class_2 = self.apply_zero_mean_pair(rotation_y_class_1, rotation_y_class_2)
        rotation_z_class_1, rotation_z_class_2 = self.apply_zero_mean_pair(rotation_z_class_1, rotation_z_class_2)
        print("Rotation Class 1", rotation_x_class_1.iloc[0, :])
        print("Rotation Class 2", rotation_x_class_2.iloc[0, :])
        # rotation_z_class_2 = self.apply_zero_mean(rotation_z_class_2)
        # print("shape", rotation_x_class_1[:,0].shape)
        # Construct dataset
        start_indices_class1 = start_indices_class1.tolist()
        start_indices_class2 = start_indices_class2.tolist()
        rotation_class_1 = self.construct_dataset(start_indices_class1, rotation_x_class_1, rotation_y_class_1, rotation_z_class_1, sequence_length=self.t)
        # print("Rotation Class 1", rotation_class_1[0][0])
        # print(rotation_class_1.shape)
        rotation_class_2 = self.construct_dataset(start_indices_class2, rotation_x_class_2, rotation_y_class_2, rotation_z_class_2, sequence_length=self.t)
        # print("Rotation Class 2", rotation_class_2[0][0])
        # print(rotation_class_2.shape)
        return rotation_class_1, rotation_class_2
    
class RotationDataset(Dataset):
    def __init__(self, class_1, class_2, x_column, y_column, z_column):
        """
        Args:
            class_1 (np.ndarray or torch.Tensor): Dataset for class 1 with shape (50, 100, 12)
            class_2 (np.ndarray or torch.Tensor): Dataset for class 2 with shape (50, 100, 12)
        """
        class_1 = class_1[:, :, [x_column, y_column, z_column]]
        class_2 = class_2[:, :, [x_column, y_column, z_column]]
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


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Adjust for bidirectional LSTM

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # Concatenate the last hidden state of both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Shape: [batch_size, hidden_dim * 2]
        logits = self.fc(hidden)  # Pass through fully connected layer
        return logits

# class RNNClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_classes):
#         super(RNNClassifier, self).__init__()
#         self.rnn = nn.RNN(
#             input_dim, 
#             hidden_dim, 
#             batch_first=True, 
#             nonlinearity='relu', 
#             dropout=0.2, 
#             num_layers=2, 
#             bidirectional=True
#         )
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Adjust for bidirectional (hidden_dim * 2)

#     def forward(self, x):
#         _, hidden = self.rnn(x)  # hidden: [num_layers * num_directions, batch_size, hidden_dim]
#         # Concatenate the last hidden state of both directions
#         hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Shape: [batch_size, hidden_dim * 2]
#         logits = self.fc(hidden)  # Pass through fully connected layer
#         return logits