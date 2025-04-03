#Preprocessing
import pandas as pd
# #load dataset
sales_data=pd.read_csv("C:\\Users\\ishit\\python\\numpy_pandas\\ca2\\Candy_Sales.csv")
#  #1. Convert date columns to datetime
sales_data["Order Date"]=pd.to_datetime(sales_data["Order Date"])
sales_data["Ship Date"]=pd.to_datetime(sales_data["Ship Date"])
# #Missing values in each col
# print(sales_data.isnull().sum())
# #2. Handling Missing Values
sales_data.fillna({"Postal Code":"Unknown","Sales":sales_data["Sales"].mean()},inplace=True)
# #3. Remove Duplicates
sales_data.drop_duplicates(inplace=True)
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

#obj3-- "Analyzing Sales Performance and Profitability Trends of Candy Products Using Data Visualization Techniques"
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# 1. sales trends over time
# plt.figure(figsize=(10,6))
# sns.lineplot(data=sales_data,x="Order Date",y="Sales")
# plt.title("Sales Trends Over Time")
# plt.xlabel("Order Date")
# plt.ylabel("Total Sales")
# plt.show()
#2. Compare sales and gross profit across regions and divisions
# plt.figure(figsize=(10,6))
# region=sales_data.groupby(["Country/Region","Division"])[["Sales","Gross Profit"]].sum().reset_index()
# melted_data=region.melt(id_vars=["Country/Region","Division"],value_vars=["Sales","Gross Profit"],var_name="Metric",value_name="Amount")
# sns.barplot(data=melted_data,x="Country/Region",y="Amount",hue="Division",errorbar=None)
# print(melted_data.head(10))
# plt.title("Sales and Gross Profit by Regions and Divisions")
# plt.xlabel("Region")
# plt.ylabel("Total Amount")
# plt.legend(title="Metric")
# plt.grid(axis="y",linestyle="--")
# plt.show()
#3. Analyze the distribution of sales and costs
# plt.figure(figsize=(12,5))
# sns.histplot(sales_data["Sales"],kde=True,bins=30,color="b",label="Sales")
# sns.histplot(sales_data["Cost"],kde=True,bins=30,color="r",label="Cost")
# plt.title("Distribution of Sales and Costs")
# plt.xlabel("Metric")
# plt.ylabel("Amount")
# plt.legend()
# plt.show()
# melted_data = sales_data.melt(value_vars=["Sales", "Cost"], var_name="Metric", value_name="Amount")
# plt.figure(figsize=(8,6))
# sns.boxplot(data=melted_data,x="Metric",y="Amount")
# plt.title("Sales vs Costs Distribution")
# plt.xlabel("Metric")
# plt.ylabel("Amount")
# plt.show()
#4. identify correlations between sales, cost, and profit
# data=sales_data[["Sales","Cost","Gross Profit"]]
# plt.figure(figsize=(10,6))
# sns.heatmap(data.corr(),annot=True,fmt=".2f",linewidths=0.5,cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

