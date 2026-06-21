import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')

df['Order Date'] = pd.to_datetime(df['Order Date'])

monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()

monthly_sales = monthly_sales.reset_index()
monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)

monthly_sales['Month_Number'] = range(1, len(monthly_sales)+1)

X = monthly_sales[['Month_Number']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X, y)

future_months = pd.DataFrame({
    'Month_Number': range(len(monthly_sales)+1, len(monthly_sales)+13)
})

future_predictions = model.predict(future_months)

prediction_df = pd.DataFrame({
    'Future Month': range(1, 13),
    'Predicted Sales': future_predictions
})

print("\nFuture Sales Forecast:")
print(prediction_df)

plt.figure(figsize=(10,6))

plt.plot(monthly_sales['Month_Number'], y, label='Historical Sales')

plt.plot(
    future_months['Month_Number'],
    future_predictions,
    label='Predicted Sales'
)

plt.xlabel('Month Number')
plt.ylabel('Sales')
plt.title('Sales Forecast Using Linear Regression')
plt.legend()

plt.savefig(r"D:\Internship Projects Thirex\Task3_Predictive_Analytics\sales_forecast.png")

plt.show()

prediction_df.to_csv(r"D:\Internship Projects Thirex\Task3_Predictive_Analytics\sales_predictions.csv", index=False)

print("\nFiles Generated Successfully:")
print("1. sales_forecast.png")
print("2. sales_predictions.csv")