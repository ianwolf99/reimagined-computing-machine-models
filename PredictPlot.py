import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set up headless Chrome WebDriver
#chrome_options = Options()
#chrome_options.add_argument("--headless")
#driver = webdriver.Chrome(options=chrome_options)
# Set up headless Chrome WebDriver
# Set up headless Chrome WebDriver
import os
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
# Set up headless Chrome WebDriver
chrome_driver_path = r'C:\Users\Sana\Downloads\HWpy\chromedriver-win64\chromedriver-win64\chromedriver.exe'  # Specify the path to chromedriver executable
chrome_options = Options()
chrome_options.add_argument(f"user-agent={user_agent}")


# Set the PATH environment variable to include the directory of chromedriver
os.environ["PATH"] = f'{os.path.dirname(chrome_driver_path)};{os.environ["PATH"]}'

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(options=chrome_options)

# Web scraping
url = "login url "
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ...
#xpath = "/html/body/app-root/app-game/div/div[1]/div[2]/div/div[2]/div[1]/app-stats-widget/div/app-stats-dropdown"
xpath = "/html/body/app-root/app-game/div/div[1]/div[2]/div/div[2]/div[1]/app-stats-widget/div/app-stats-dropdown/div/div[2]"

try:
    driver.get(url)

    # Wait for the page to load completely
    wait = WebDriverWait(driver, 10)  # Adjust the timeout as needed
    element = wait.until(EC.visibility_of_element_located((By.XPATH, xpath)))

    round_history_data = element.text
    print(round_history_data)

except Exception as e:
    print(f"Error during web scraping: {str(e)}")
    driver.quit()
    exit(1)


# Close the WebDriver
driver.quit()

# Data processing with pandas
round_history_list = [int(num) for num in round_history_data.split()]
df = pd.DataFrame({'Round History': round_history_list})

# Feature engineering
df['Previous_Round'] = df['Round History'].shift(1)
df.dropna(inplace=True)

# Creating lag features to capture historical trends
for lag in range(1, 11):
    df[f'Previous_Round_{lag}'] = df['Round History'].shift(lag)

# Prepare the target and features
X = df.drop(columns=['Round History'])
y = df['Round History']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine Learning - Random Forest Regression
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the next round history
latest_data = np.array(df.iloc[-1][1:]).reshape(1, -1)
next_round = model.predict(latest_data)[0]
print(f"Predicted Next Round: {next_round}")

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Testing Data: {mse}")

# Save the round history to an Excel file with a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
excel_filename = f"round_history_{timestamp}.xlsx"
df.to_excel(excel_filename, index=False)

# Plot responsive line chart
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Round History'], label='Actual Round History')
plt.xlabel('Timestamp')
plt.ylabel('Round History')
plt.title('Round History Over Time')
plt.legend()
plt.grid(True)

# Save the chart as an image
chart_filename = f"round_history_chart_{timestamp}.png"
plt.savefig(chart_filename)
plt.show()
