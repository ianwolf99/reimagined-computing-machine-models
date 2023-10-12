import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import numpy as np

# Read the HTML file
with open('file.html', 'r') as file:
    html_content = file.read()

# Define a regular expression pattern to match the desired lines
pattern = r'color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\);">(.*?)x'

# Find all matches in the HTML content
matches = re.findall(pattern, html_content)

# Create a mapping of the specific RGB values to color names
rgb_to_color_name = {
    (145, 62, 248): 'Purple',
    (52, 180, 255): 'Blue',
    (192, 23, 180): 'Magenta',
}

# Create a list of dictionaries to store the data
data = []

for match in matches:
    rgb_value = (int(match[0]), int(match[1]), int(match[2]))
    color_name = rgb_to_color_name.get(rgb_value, 'Unknown Color')
    multiplier = float(match[3])
    data.append({'Color': color_name, 'Multiplier': multiplier})

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Define a function to predict the next 20 values for a given color using XGBoost
def predict_next_20_values_xgboost(color_data):
    features = [f'Multiplier_Lag_{lag}' for lag in range(1, 6)]

    historical_indices = color_data.index
    last_index = historical_indices.max()
    next_indices = np.arange(last_index + 2, last_index + 42, 2)  # Predict next 20 values
    
    # Reindex the color_data DataFrame to include all the required indices
    color_data = color_data.reindex(next_indices, fill_value=None)
    
    next_features = color_data[features]

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(color_data[features], color_data['Multiplier'])

    next_predictions = xgb_model.predict(next_features)
    return next_indices, next_predictions


# Create lagged variables
for lag in range(1, 6):  # You can adjust the lag window
    df[f'Multiplier_Lag_{lag}'] = df['Multiplier'].shift(lag)

# Train models and predict next values for each color using XGBoost
colors = ['Purple', 'Blue', 'Magenta']
predictions = {'XGBoost': {}}

for color in colors:
    color_data = df[df['Color'] == color]
    indices, preds = predict_next_20_values_xgboost(color_data)
    predictions['XGBoost'][color] = {'Indices': indices, 'Predictions': preds}

# Plot the historical and predicted data
plt.figure(figsize=(10, 6))
line_colors = ['purple', 'blue', 'magenta']
line_styles = ['-', '--']

for i, color in enumerate(colors):
    for model_name, linestyle in zip(predictions.keys(), line_styles):
        color_data = df[df['Color'] == color]
        plt.plot(color_data.index, color_data['Multiplier'], label=f'Actual {color} ({model_name})',
                 linestyle=linestyle, color=line_colors[i])

        indices = predictions[model_name][color]['Indices']
        preds = predictions[model_name][color]['Predictions']
        plt.plot(indices, preds, label=f'Predicted {color} ({model_name})', linestyle=linestyle, color=line_colors[i])

plt.xlabel('Index')
plt.ylabel('Multiplier')
plt.title('Multiplier vs. Color with Predictions')
plt.legend()
plt.grid(True)

# Set the y-axis limits and adjust y-axis ticks to have a step of 2
plt.ylim(0, 100)  # You can adjust the range as needed
plt.yticks(np.arange(0, 102, 2))  # Adjust y-axis ticks

# Set the x-axis ticks to have a step of 2
plt.xticks(np.arange(0, len(df) + 21, 2))  # Include predicted values

plt.show()
