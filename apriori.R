# Apriori

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('...../Apriori-R/Apriori/Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('/......./Apriori-R/Apriori/Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
