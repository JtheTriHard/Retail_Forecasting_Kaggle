# Retail_Forecasting_Kaggle

View progress here: https://trello.com/b/3RArR6un/retail-forecasting  
View the example Tableau visualizations here (in progress): https://public.tableau.com/profile/joey.gronovius#!/vizhome/Retail_Forecasting/MainStory
## Contents
### SQL
Database_Builder.sql - script used to build MySQL relational database on a personal server.  
Queries.sql - script used to pull different datasets from the database.  
![alt text](https://github.com/JtheTriHard/Retail_Forecasting_Kaggle/blob/master/SQL/DB_Diagram.png)

### Python
Data_Processing.py - script used to clean kaggle dataset for Tableau and forecasting models.  
Model_MLP_Embedding.py - TensorFlow multilayer perceptron neural network utilizing entity embeddings for shop & item IDs. (Suboptimal)  
Model_MLP_TargetEncoded.py - Uses Target Encoding instead. Currently the optimal model (MSE 0.0076)
