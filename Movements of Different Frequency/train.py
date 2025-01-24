from data_prep import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# # Function for Leave-One-Out Validation (LOOV)
# def loov_evaluation(dataset, model_class, input_dim, hidden_dim, num_classes, num_epochs, learning_rate, device):
#     """
#     Performs Leave-One-Out Validation with randomly chosen samples from each class as the test set
#     and calculates average time-step accuracies for multiple iterations.

#     Args:
#         dataset (Dataset): The entire dataset.
#         model_class (nn.Module): The model class to be instantiated.
#         input_dim (int): Input dimension of the data (e.g., 3 for Rx, Ry, Rz).
#         hidden_dim (int): Hidden layer size.
#         num_classes (int): Number of output classes.
#         num_epochs (int): Number of epochs for training.
#         learning_rate (float): Learning rate for the optimizer.
#         device (torch.device): Device to run the computation on.

#     Returns:
#         np.ndarray: Average time-step accuracies across all iterations.
#     """
#     # Split the dataset into class 1 and class 2
#     class_1_indices = [i for i in range(len(dataset)) if dataset[i][1] == 0]
#     class_2_indices = [i for i in range(len(dataset)) if dataset[i][1] == 1]

#     total_iterations = 20  # Number of LOOV iterations
#     sequence_length = 100  # Assume sequence length is 100
#     time_step_accuracies = np.zeros((total_iterations, sequence_length))  # [iterations, time steps]

#     for iteration in tqdm(range(total_iterations), desc="LOOV Iterations"):
#         # Randomly select 1 sample from each class for the test set
#         test_class_1 = random.sample(class_1_indices, 1)
#         test_class_2 = random.sample(class_2_indices, 1)
#         test_indices = test_class_1 + test_class_2

#         # Use the rest for training
#         train_indices = list(set(range(len(dataset))) - set(test_indices))

#         # Create train and test datasets
#         train_dataset = Subset(dataset, train_indices)
#         test_dataset = Subset(dataset, test_indices)

#         # DataLoaders
#         train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

#         # Initialize model
#         model = model_class(input_dim, hidden_dim, num_classes).to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#         # Train the model
#         model.train()
#         epoch_losses = []
#         for epoch in range(num_epochs):
#             epoch_loss = 0
#             for batch_data, batch_labels in train_loader:
#                 batch_data = batch_data.to(device)
#                 batch_labels = batch_labels.float().to(device)  # Use float and match dimensions for BCELoss

#                 optimizer.zero_grad()
#                 outputs = model(batch_data)
#                 outputs = torch.sigmoid(outputs)  # Ensure outputs are in [0, 1] range for BCELoss
#                 loss = criterion(outputs, batch_labels)
#                 loss.backward()
#                 optimizer.step()

#                 epoch_loss += loss.item()

#             epoch_losses.append(epoch_loss / len(train_loader))


#         # Evaluate the model
#         model.eval()
#         with torch.no_grad():
#             for batch_data, batch_labels in test_loader:
#                 batch_data = batch_data.to(device)
#                 batch_labels = batch_labels.float().to(device)  # Match dimensions for BCELoss

#                 for t in range(batch_data.size(1)):  # Iterate over sequence_length (100)
#                     step_input = batch_data[:, :t+1, :]  # Use data up to time step t

#                     step_output, _ = model.rnn(step_input)  # RNN output: [batch_size, t+1, hidden_dim]
#                     final_output = step_output[:, -1, :]  # Last time step's output
#                     logits = model.fc(final_output)  # [batch_size, num_classes]
#                     logits = torch.sigmoid(logits)  # Apply sigmoid to logits

#                     step_pred = (logits > 0.5).long()  # Binarize predictions
#                     correct = (step_pred == batch_labels).sum().item()
#                     time_step_accuracies[iteration, t] += correct / batch_labels.size(0)  # Normalize by batch size

#         print(f"Iteration {iteration+1}/{total_iterations} completed.")

#     # Average accuracies across all iterations for each time step
#     average_time_step_accuracies = np.mean(time_step_accuracies, axis=0)

#     # Print final average time-step accuracies
#     print(f"Average Time-Step Accuracies: {average_time_step_accuracies}")

#     return average_time_step_accuracies
# if __name__ == '__main__':
#     print("PyTorch Version: ", torch.__version__)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     # Paths
#     data_folder_1 = r'C:\Users\11424\Documents\phd\pytorch_experiment_nc\Movements of Different Frequency\pwm_37'
#     data_folder_2 = r'C:\Users\11424\Documents\phd\pytorch_experiment_nc\Movements of Different Frequency\pwm_40'

#     # Data processing
#     t = 100
#     data_processor = DataProcessor(data_folder_1, data_folder_2, t)
#     rotation_class_1, rotation_class_2 = data_processor.build()
#     print(f"Rotation class 1 shape: {rotation_class_1.shape}")
#     print("Sample from rotation class 1:", rotation_class_1[0])
#     print(f"Rotation class 2 shape: {rotation_class_2.shape}")


#     # Sensor configurations
#     sensors = {
#         "Sensor from port 10": (0, 4, 8),  # Indices for Rx, Ry, Rz of Sensor 1
#         "Sensor from port 11": (1, 5, 9),  # Indices for Rx, Ry, Rz of Sensor 2
#         "Sensor from port 12": (2, 6, 10),  # Indices for Rx, Ry, Rz of Sensor 3
#         "Sensor from port 13": (3, 7, 11)  # Indices for Rx, Ry, Rz of Sensor 4
#     }

#     # Model configurations
#     input_dim = 3  # Number of features (Rx, Ry, Rz)
#     hidden_dim = 64  # Number of hidden units
#     num_classes = 2  # Number of classes
#     num_epochs = 30  # Number of training epochs
#     learning_rate = 0.001  # Learning rate

#     dataset_sensor_1 = RotationDataset(rotation_class_1, rotation_class_2, 0, 4, 8)
#     # print(f"Length of dataset_sensor_1: {len(dataset_sensor_1)}")
#     # print(f"Sample from dataset_sensor_1: {dataset_sensor_1[0][0].size()}")

#     # Run LOOV for each sensor
#     sensor_accuracies = {}

#     for sensor_name, (x_idx, y_idx, z_idx) in sensors.items():
#         print(f"Processing {sensor_name}...")

#         # Create dataset for the specific sensor
#         dataset_sensor = RotationDataset(rotation_class_1, rotation_class_2, x_idx, y_idx, z_idx)

#         # Perform LOOV evaluation
#         average_time_step_accuracies = loov_evaluation(
#             dataset_sensor, RNNClassifier, input_dim, hidden_dim, num_classes, num_epochs, learning_rate, device
#         )

#         # Store results
#         sensor_accuracies[sensor_name] = average_time_step_accuracies

#     # Plot results for all sensors
#     import matplotlib.pyplot as plt
#     import pandas as pd

#     # Convert sensor accuracies to a DataFrame
#     sensor_accuracies_df = pd.DataFrame(
#         sensor_accuracies,
#         index=range(1, t + 1)  # Time steps as row indices
#     ).T  # Transpose to match shape: 4 (sensors) x 100 (time steps)

#     # Save to CSV
#     output_file = f"sensor_accuracies_{data_folder_1[-2:0]}{data_folder_2[-2:0]}.csv"
#     sensor_accuracies_df.to_csv(output_file, index=True, header=True)
#     print(f"Sensor accuracies saved to {output_file}")

#     plt.figure(figsize=(12, 8))
#     for sensor_name, accuracies in sensor_accuracies.items():
#         plt.plot(range(1, t + 1), accuracies, label=sensor_name, marker='o')

#     plt.title("Average Time-Step Accuracies Across Sensors")
#     plt.xlabel("Time Step")
#     plt.ylabel("Accuracy (%)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def loov_evaluation(dataset, model_class, input_dim, hidden_dim, num_classes, num_epochs, learning_rate, device):
    # Split dataset into class indices
    class_1_indices = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    class_2_indices = [i for i in range(len(dataset)) if dataset[i][1] == 1]

    total_iterations = 100
    sequence_length = 100
    time_step_accuracies = np.zeros((total_iterations, sequence_length))

    for iteration in tqdm(range(total_iterations), desc="LOOV Iterations"):
        # Select test indices
        test_class_1 = random.sample(class_1_indices, 1)
        test_class_2 = random.sample(class_2_indices, 1)
        test_indices = test_class_1 + test_class_2

        # Train/Test split
        train_indices = list(set(range(len(dataset))) - set(test_indices))
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        # Initialize model
        model = model_class(input_dim, hidden_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.long().to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

        # Evaluation loop
        model.eval()
        with torch.no_grad():
            step_sample_counts = np.zeros(sequence_length)
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.long().to(device)
                for t in range(batch_data.size(1)):
                    step_input = batch_data[:, :t+1, :]
                    step_output, _ = model.lstm(step_input)
                    final_output = step_output[:, -1, :]
                    logits = model.fc(final_output)
                    step_pred = torch.argmax(logits, dim=1)
                    correct = (step_pred == batch_labels).float().sum().item()
                    time_step_accuracies[iteration, t] += correct
                    step_sample_counts[t] += batch_labels.size(0)
            
            # Normalize accuracies
            step_sample_counts[step_sample_counts == 0] = 1  # Avoid division by zero
            time_step_accuracies[iteration, :] /= step_sample_counts

    average_time_step_accuracies = np.mean(time_step_accuracies, axis=0)
    print(f"Final Average Time-Step Accuracies: {average_time_step_accuracies}")
    return average_time_step_accuracies

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    data_folder_1 = r'C:\Users\11424\Documents\phd\pytorch_experiment_nc\Movements of Different Frequency\pwm_37'
    data_folder_2 = r'C:\Users\11424\Documents\phd\pytorch_experiment_nc\Movements of Different Frequency\pwm_40'

    # Data processing
    t = 100
    data_processor = DataProcessor(data_folder_1, data_folder_2, t)
    rotation_class_1, rotation_class_2 = data_processor.build()
    print(f"Rotation class 1 shape: {rotation_class_1.shape}")
    print("Sample from rotation class 1:", rotation_class_1[0])
    print(f"Rotation class 2 shape: {rotation_class_2.shape}")

    # Sensor configurations
    sensors = {
        "Sensor from port 10": (0, 4, 8),  # Indices for Rx, Ry, Rz of Sensor 1
        "Sensor from port 11": (1, 5, 9),  # Indices for Rx, Ry, Rz of Sensor 2
        "Sensor from port 12": (2, 6, 10),  # Indices for Rx, Ry, Rz of Sensor 3
        "Sensor from port 13": (3, 7, 11)  # Indices for Rx, Ry, Rz of Sensor 4
    }

    # Model configurations
    input_dim = 3  # Number of features (Rx, Ry, Rz)
    hidden_dim = 128  # Number of hidden units
    num_classes = 2  # Number of classes
    num_epochs = 30  # Number of training epochs
    learning_rate = 0.005  # Learning rate

    dataset_sensor_1 = RotationDataset(rotation_class_1, rotation_class_2, 0, 4, 8)

    # Run LOOV for each sensor
    sensor_accuracies = {}

    for sensor_name, (x_idx, y_idx, z_idx) in sensors.items():
        print(f"Processing {sensor_name}...")

        # Create dataset for the specific sensor
        dataset_sensor = RotationDataset(rotation_class_1, rotation_class_2, x_idx, y_idx, z_idx)

        # Perform LOOV evaluation
        average_time_step_accuracies = loov_evaluation(
            dataset_sensor, LSTMClassifier, input_dim, hidden_dim, num_classes, num_epochs, learning_rate, device
        )

        # Store results
        sensor_accuracies[sensor_name] = average_time_step_accuracies

    # Plot results for all sensors
    import matplotlib.pyplot as plt
    import pandas as pd

    # Convert sensor accuracies to a DataFrame
    sensor_accuracies_df = pd.DataFrame(
        sensor_accuracies,
        index=range(1, t + 1)  # Time steps as row indices
    ).T  # Transpose to match shape: 4 (sensors) x 100 (time steps)

    # Save to CSV
    output_file = f"sensor_accuracies_{data_folder_1[-2:]}_{data_folder_2[-2:]}.csv"
    sensor_accuracies_df.to_csv(output_file, index=True, header=True)
    print(f"Sensor accuracies saved to {output_file}")

    plt.figure(figsize=(12, 8))
    for sensor_name, accuracies in sensor_accuracies.items():
        plt.plot(range(1, t + 1), accuracies, label=sensor_name, marker='o')

    plt.title("Average Time-Step Accuracies Across Sensors")
    plt.xlabel("Time Step")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
