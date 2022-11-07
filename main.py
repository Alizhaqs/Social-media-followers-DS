import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from autots import AutoTS

data = pd.read_csv("stats.csv")
data.drop(data.tail(1).index, inplace=True)
data.head()

plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Total Followers At The End of Every Month")
sns.barplot(x="followers_total", y="period_end", data=data)
plt.show()

model = AutoTS(forecast_length=4, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='period_end', value_col='followers_gained', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)