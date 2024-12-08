import os
import re
import pandas as pd

# Function to read the configuration file and get the path to the reports
def read_config():
    config_file_path = './Config/Configuration.txt'
    path_to_reports = None
    with open(config_file_path, 'r') as file:
        for line in file:
            if line.startswith('PATH_TO_MODEL_REPORT'):
                path_to_reports = line.split('=')[1].strip()
                break
    return path_to_reports

# Directory containing the reports (read from the configuration file)
reports_directory = read_config()

# Check if the reports directory was found
if not reports_directory or not os.path.exists(reports_directory):
    print(f"Error: The reports directory '{reports_directory}' is invalid or not found.")
    exit()

# Regular expressions for accuracy, model names, and dataset names
accuracy_pattern = r"Accuracy: ([0-9.]+)"
model_pattern = r"(.*?) Report for"
dataset_pattern = r"Report for (.*?),"

# Store the results
results = []

# Process each report file in subdirectories
for subdir, _, files in os.walk(reports_directory):
    for report_file in files:
        if report_file.endswith('.txt'):  # Only process .txt files
            file_path = os.path.join(subdir, report_file)
            
            with open(file_path, 'r') as file:
                content = file.read()
                
                # Extract accuracies, models, and dataset names
                accuracies = re.findall(accuracy_pattern, content)
                models = re.findall(model_pattern, content)
                dataset_names = re.findall(dataset_pattern, content)
                
                if accuracies:
                    # Convert accuracies to float
                    accuracies = [float(acc) for acc in accuracies]
                    
                    # Find the highest accuracy in this file
                    highest_accuracy = max(accuracies)
                    best_model_index = accuracies.index(highest_accuracy)
                    best_model = models[best_model_index]
                    dataset_name = dataset_names[best_model_index]
                    
                    # Determine the attack type (combined, fuzzy, replay)
                    if 'Combined' in subdir:
                        attack_type = 'Combined'
                    elif 'Fuzzy' in subdir:
                        attack_type = 'Fuzzy'
                    elif 'Replay' in subdir:
                        attack_type = 'Replay'
                    else:
                        attack_type = 'Unknown'
                    
                    # Store the result
                    results.append([attack_type, best_model, highest_accuracy, dataset_name, file_path])

# Create a DataFrame to summarize the results
df = pd.DataFrame(results, columns=['Attack Type', 'Best Model', 'Highest Accuracy', 'Dataset Name', 'Report File'])

# Ensure the Evaluation directory exists
if not os.path.exists('./Evaluation'):
    os.makedirs('./Evaluation')

# Save model performance summary to CSV inside the Evaluation directory
df.to_csv('./Evaluation/Model_performance_summary.csv', index=False)

# Count how many times each model is the best for each attack type
model_counts = df.groupby(['Attack Type', 'Best Model']).size().reset_index(name='Count')

# Find the best model for each attack type (the one with the highest count)
best_models = model_counts.loc[model_counts.groupby('Attack Type')['Count'].idxmax()]

# Save the summary of best models for each attack type inside the Evaluation directory
best_models.to_csv('./Evaluation/Best_models_per_attack_count.csv', index=False)

print("Model performance summary saved to './Evaluation/Model_performance_summary.csv'.")
print("Best models per attack type based on count saved to './Evaluation/Best_models_per_attack.csv'.")
