#Preprocessing
# import pandas as pd
# #load dataset
# sales_data=pd.read_csv("C:\\Users\\ishit\\python\\numpy_pandas\\ca2\\Candy_Sales.csv")
#  #1. Convert date columns to datetime
# sales_data["Order Date"]=pd.to_datetime(sales_data["Order Date"])
# sales_data["Ship Date"]=pd.to_datetime(sales_data["Ship Date"])
# #Missing values in each col
# print(sales_data.isnull().sum())
# #2. Handling Missing Values
# sales_data.fillna({"Postal Code":"Unknown","Sales":sales_data["Sales"].mean()},inplace=True)
# #3. Remove Duplicates
# sales_data.drop_duplicates(inplace=True)
# #4. Dataset Summary
# print(sales_data.info())
# print(sales_data.head())

# obj1
#1. calculate the total revenue, average revenue per order, and standard deviation of sales from the Candy_Sales dataset.
# import numpy as np
# import pandas as pd
# #load dataset
# sales_data=pd.read_csv("C:\\Users\\ishit\\python\\numpy_pandas\\ca2\\Candy_Sales.csv")
# sales=np.array(sales_data["Sales"])
# total_revenue=np.sum(sales)
# average_revenue=np.mean(sales)
# std_dev_sales=np.std(sales)
# print("Total Revenue:",total_revenue)
# print("Average Revenue per Order:",average_revenue)
# print("Standard Deviation of Sales:",std_dev_sales)

# obj2
#find the total number of orders, total units sold, and the state with the highest sales
# import pandas as pd
# sales_data=pd.read_csv("C:\\Users\\ishit\\python\\numpy_pandas\\ca2\\Candy_Sales.csv")
# total_orders=sales_data.shape[0]
# total_units=sales_data["Units"].sum()
# state_highest_sale=sales_data.groupby("State/Province")["Sales"].sum().idxmax()
# print("Total number of Orders:",total_orders)
# print("Total Units sold:",total_units)
# print("State with the highest Sales:",state_highest_sale)

#obj3

