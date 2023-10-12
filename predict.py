import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

chrome_driver_path = '\chromedriver-win64\chromedriver-win64\chromedriver.exe'  # Specify the path to chromedriver executable
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)

# Web scraping
url = "https://www.betika.com/en-ke/aviator"
try:
    driver.get(url)
    time.sleep(5)  # Wait for the page to load (adjust as needed)

    # Find the element with XPath
    xpath = "/html/body/app-root/app-game/div/div[1]/div[2]/div/div[2]/div[1]/app-stats-widget/div/app-stats-dropdown"
    element = driver.find_element_by_xpath(xpath)
    round_history_data = element.text

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

