import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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
print(df)

# Split the data for each color
purple_data = df[df['Color'] == 'Purple']
blue_data = df[df['Color'] == 'Blue']
magenta_data = df[df['Color'] == 'Magenta']

# Define a function to predict the next 10 indexes using the specified model
def predict_next_10_indexes(data, model):
    X = data.index.values.reshape(-1, 1)
    y = data['Multiplier']
    model.fit(X, y)
    next_indexes = range(data.index.max() + 1, data.index.max() + 11)
    next_predictions = model.predict(list(next_indexes))
    return next_indexes, next_predictions

# Train and make predictions for each color
rf_model = RandomForestRegressor(n_estimators=100)
lr_model = LinearRegression()

purple_next_indexes, purple_next_predictions = predict_next_10_indexes(purple_data, rf_model)
blue_next_indexes, blue_next_predictions = predict_next_10_indexes(blue_data, lr_model)
magenta_next_indexes, magenta_next_predictions = predict_next_10_indexes(magenta_data, lr_model)

# Plot line graphs for the three colors with specified line colors and styles
plt.plot(purple_data.index, purple_data['Multiplier'], label='Purple', color='purple')
plt.plot(blue_data.index, blue_data['Multiplier'], label='Blue', color='blue')
plt.plot(magenta_data.index, magenta_data['Multiplier'], label='Magenta', color='magenta')

# Plot predicted lines with dotted style
plt.plot(purple_next_indexes, purple_next_predictions, label='Purple Prediction', linestyle='dotted', color='purple')
plt.plot(blue_next_indexes, blue_next_predictions, label='Blue Prediction', linestyle='dotted', color='blue')
plt.plot(magenta_next_indexes, magenta_next_predictions, label='Magenta Prediction', linestyle='dotted', color='magenta')

plt.xlabel('Index')
plt.ylabel('Multiplier')
plt.title('Multiplier vs. Color with Predictions')
plt.legend()
plt.show()
