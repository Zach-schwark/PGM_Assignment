from ucimlrepo import fetch_ucirepo 
import pandas
  
print("fecthing data")
# fetch dataset 
polish_companies_bankruptcy = fetch_ucirepo(id=365) 

X: pandas.DataFrame
y: pandas.DataFrame
  
# data (as pandas dataframes) 
X = polish_companies_bankruptcy.data.features 
y = polish_companies_bankruptcy.data.targets 

print("fecthed")
  
# metadata 
print(polish_companies_bankruptcy.metadata) 
  
# variable information 
print(polish_companies_bankruptcy.variables) 

print(y)
print(X)
