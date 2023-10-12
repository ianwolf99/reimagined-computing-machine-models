import os
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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up headful Chrome WebDriver
chrome_driver_path = r'C:\Users\Sana\Downloads\HWpy\chromedriver-win64\chromedriver-win64\chromedriver.exe'
profile_path = "C:\\Users\\Sana\\AppData\\Local\\Google\\Chrome\\User Data\\Profile"
chrome_options = Options()
chrome_options.add_argument(f"--user-data-dir={profile_path}")
chrome_options.add_argument("--start-maximized")  # Maximize the browser window
chrome_options.add_argument(f"executable_path={chrome_driver_path}")  # Specify the path to chromedriver

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(options=chrome_options)

# Rest of your code...



# Open the login page
# Open the login page
login_url = "https://www.betika.com/en-ke/login?next=%2Faviator"
print(f"Opening login page: {login_url}")
driver.get(login_url)

# Allow the user to input the XPath for the round history element
round_history_xpath = input("Please enter the XPath for the round history element: ")
print(f"Using XPath for round history element: {round_history_xpath}")

try:
    # Wait for the page to load after login
    wait = WebDriverWait(driver, 20)
    round_history_element = wait.until(EC.presence_of_element_located((By.XPATH, round_history_xpath)))

    round_history_data = round_history_element.text
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
