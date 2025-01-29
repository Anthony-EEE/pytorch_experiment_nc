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
# Fix random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism but might slow down training

# Leave-One-Out Validation (LOOV) Evaluation
def loov_evaluation(dataset, model_class, input_dim, hidden_dim, num_classes, num_epochs, learning_rate, device):
    class_1_indices = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    class_2_indices = [i for i in range(len(dataset)) if dataset[i][1] == 1]

    total_iterations = 100
    sequence_length = 100
    time_step_accuracies = np.zeros((total_iterations, sequence_length))

    for iteration in tqdm(range(total_iterations), desc="LOOV Iterations"):
        test_class_1 = random.sample(class_1_indices, 1)
        test_class_2 = random.sample(class_2_indices, 1)
        test_indices = test_class_1 + test_class_2
        train_indices = list(set(range(len(dataset))) - set(test_indices))
        
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        # Initialize model
        model = model_class(input_dim, hidden_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.long().to(device)
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
                batch_data, batch_labels = batch_data.to(device), batch_labels.long().to(device)
                for t in range(batch_data.size(1)):
                    step_input = batch_data[:, :t+1, :]
                    step_output, _ = model.lstm(step_input)
                    final_output = step_output[:, -1, :]
                    logits = model.fc(final_output)
                    step_pred = torch.argmax(logits, dim=1)
                    correct = (step_pred == batch_labels).float().sum().item()
                    time_step_accuracies[iteration, t] += correct
                    step_sample_counts[t] += batch_labels.size(0)
            
            step_sample_counts[step_sample_counts == 0] = 1  # Avoid division by zero
            time_step_accuracies[iteration, :] /= step_sample_counts

    average_time_step_accuracies = np.mean(time_step_accuracies, axis=0)
    print(f"Final Average Time-Step Accuracies: {average_time_step_accuracies}")
    return average_time_step_accuracies

if __name__ == '__main__':
    set_seed(42)  # Set a fixed seed for reproducibility
    print("PyTorch Version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data paths
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
        "Sensor from port 10": (0, 4, 8),
        "Sensor from port 11": (1, 5, 9),
        "Sensor from port 12": (2, 6, 10),
        "Sensor from port 13": (3, 7, 11)
    }

    # Model configurations
    input_dim = 3
    hidden_dim = 64
    num_classes = 2
    num_epochs = 100
    learning_rate = 0.005

    sensor_accuracies = {}

    # Run LOOV for each sensor
    for sensor_name, (x_idx, y_idx, z_idx) in sensors.items():
        print(f"Processing {sensor_name}...")

        dataset_sensor = RotationDataset(rotation_class_1, rotation_class_2, x_idx, y_idx, z_idx)

        average_time_step_accuracies = loov_evaluation(
            dataset_sensor, LSTMClassifier, input_dim, hidden_dim, num_classes, num_epochs, learning_rate, device
        )

        sensor_accuracies[sensor_name] = average_time_step_accuracies

    # Convert sensor accuracies to DataFrame and save
    sensor_accuracies_df = pd.DataFrame(
        sensor_accuracies, index=range(1, t + 1)
    ).T 

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