#Data Preprocessing & Cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro,probplot
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
#EDA
#1. Summary Statistics
# print("Summary stats-")
# print(sales_data.describe())
#2. Correlation & Covariance
# print("Correlation Matrix")
# print(sales_data[["Sales","Units","Gross Profit","Cost"]].corr())
# print("Covariance Matrix")
# print(sales_data[["Sales","Units","Gross Profit","Cost"]].cov())
# # obj1
#1. calculate the total revenue, average revenue per order, and standard deviation of sales from the Candy_Sales dataset.
# sales=np.array(sales_data["Sales"])
# total_revenue=np.sum(sales)
# average_revenue=np.mean(sales)
# std_dev_sales=np.std(sales)
# print("Total Revenue:",total_revenue)
# print("Average Revenue per Order:",average_revenue)
# print("Standard Deviation of Sales:",std_dev_sales)

# obj2
#find the total number of orders, total units sold, and the state with the highest sales
# total_orders=sales_data.shape[0]
# total_units=sales_data["Units"].sum()
# state_highest_sale=sales_data.groupby("State/Province")["Sales"].sum().idxmax()
# print("Total number of Orders:",total_orders)
# print("Total Units sold:",total_units)
# print("State with the highest Sales:",state_highest_sale)

#obj3-- "Analyzing Sales Performance and Profitability Trends of Candy Products Using Data Visualization Techniques"
# 1. sales trends over time
# plt.figure(figsize=(10,6))
# sns.lineplot(data=sales_data,x="Order Date",y="Sales")
# plt.title("Sales Trends Over Time")
# plt.xlabel("Order Date")
# plt.ylabel("Total Sales")
# plt.show()
#2. Compare sales and gross profit across regions and divisions
# region = sales_data.groupby(["Country/Region", "Division"])[["Sales", "Gross Profit"]].sum().reset_index()
# print(region)
# region["Region-Division"] = region["Country/Region"] + " - " + region["Division"]
# melted_data = region.melt(
#     id_vars=["Region-Division"],
#     value_vars=["Sales", "Gross Profit"],
#     var_name="Metric",
#     value_name="Amount"
# )
# plt.figure(figsize=(14, 6))
# ax = sns.barplot(data=melted_data, x="Region-Division", y="Amount", hue="Metric", palette="Set2", errorbar=None)
# for p in ax.patches:
#     height = p.get_height()
#     ax.annotate(f'{int(height)}',
#                 (p.get_x() + p.get_width() / 2., height),
#                 ha='center', va='bottom', fontsize=9, color='black')
# plt.title("Sales and Gross Profit by Division for Each Region")
# plt.xlabel("Region - Division")
# plt.ylabel("Total Amount")
# plt.xticks(rotation=45, ha="right")
# plt.grid(axis="y", linestyle="--")
# plt.tight_layout()
# plt.show()
#3. Analyze the distribution of sales and costs
# sns.set(style="whitegrid")
# plt.figure(figsize=(14, 6))
# #Sales Histogram
# plt.subplot(1, 2, 1)
# sales_plot = sns.histplot(sales_data['Sales'], bins=10, color='skyblue', kde=True)
# plt.title('Distribution of Sales')
# plt.xlabel('Sales')
# plt.ylabel('Frequency')
# for bar in sales_plot.patches:
#     height = bar.get_height()
#     if height > 0:
#         sales_plot.text(
#             bar.get_x() + bar.get_width() / 2,
#             height,
#             f'{int(height)}',
#             ha='center',
#             va='bottom',
#             fontsize=9,
#             fontweight='bold'
#         )
# plt.subplot(1, 2, 2)
# cost_plot = sns.histplot(sales_data['Cost'], bins=10, color='salmon', kde=True)
# plt.title('Distribution of Costs')
# plt.xlabel('Cost')
# plt.ylabel('Frequency')
# for bar in cost_plot.patches:
#     height = bar.get_height()
#     if height > 0:
#         cost_plot.text(
#             bar.get_x() + bar.get_width() / 2,
#             height,
#             f'{int(height)}',
#             ha='center',
#             va='bottom',
#             fontsize=9,
#             fontweight='bold'
#         )
# plt.tight_layout()
# plt.show()

#obj4
# detect and visualize outliers in the Sales and Gross Profit 
# def detect_outliers(data,col):
#     Q1=data[col].quantile(0.25)
#     Q3=data[col].quantile(0.75)
#     IQR=Q3-Q1
#     lower_bound=Q1-1.5*IQR
#     upper_bound=Q3+1.5*IQR
#     outliers=data[(data[col]<lower_bound) | (data[col]>upper_bound)]
#     return outliers
# sales_outlier=detect_outliers(sales_data,"Sales")
# gross_profit_outlier=detect_outliers(sales_data,"Gross Profit")
# print("Outliers in Sales:",sales_outlier)
# print("Outliers in Gross Profit:",gross_profit_outlier)
# #box plot
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# sns.boxplot(sales_data["Sales"],color="skyblue")
# plt.title("Box Plot of Sales")
# plt.subplot(1,2,2)
# sns.boxplot(sales_data["Gross Profit"],color="lightcoral")
# plt.title("Box Plot of Gross Profit")
# plt.tight_layout()
# plt.show()
#4. identify correlations between sales, cost, and profit
# data=sales_data[["Sales","Cost","Gross Profit"]]
# plt.figure(figsize=(10,6))
# sns.heatmap(data.corr(),annot=True,fmt=".2f",linewidths=0.5,cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()
#obj5
# To determine whether the monthly sales of a top selling product follow a normal distribution using descriptive statistics and the Shapiro-Wilk test
# top_product = sales_data['Product Name'].value_counts().idxmax()
# sales_data['Month'] = sales_data['Order Date'].dt.to_period('M')
# product_df = sales_data[sales_data['Product Name'] == top_product].copy()
# monthly_sales = sales_data.groupby('Month')['Sales'].sum()
# shapiro_stat, shapiro_p = shapiro(monthly_sales)
# desc_stats = monthly_sales.describe()
# print("Descriptive Statistics:\n", desc_stats)
# print(f"\nShapiro-Wilk Test:\nW = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
# if shapiro_p < 0.05:
#     print("Conclusion: Sales do NOT follow a normal distribution.")
# else:
#     print("Conclusion: Sales MAY follow a normal distribution.")
# # Plot histogram
# plt.figure(figsize=(10, 5))
# sns.histplot(monthly_sales, kde=True, bins=10, color='skyblue')
# plt.title(f'Monthly Sales Distribution: {top_product}')
# plt.xlabel('Sales')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # Q-Q plot
# plt.figure(figsize=(6, 6))
# probplot(monthly_sales, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Monthly Sales")
# plt.grid(True)
# plt.show()
