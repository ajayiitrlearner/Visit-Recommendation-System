# vist-recommendation-system
Recommendations System for Employee Relationship (ER) Managers: To develop a recommendation system that suggests top branches (referred to as "SolIDs" or Branch IDs) to Employee Relationship (ER) Managers based on various features and historical visit data, while considering specific constraints and ensuring the relevance of recommendations.



Components:

1. Data Preparation:
   - Training Data: Contains information about ER Managers, branch details (SolIDs), and historical visit data.
   - Active Data: Holds information about branches currently active or relevant.

2. Model Training:
   - Random Forest Classifier: Trained using features from the training data to predict the likelihood of recommending a branch (SolID).
   - Hyperparameter Tuning: Performed using RandomizedSearchCV to find the best model parameters.

3. Recommendation Generation:
   - Custom Recommendation Function (`Custom_Recommend_SOLID`): Generates recommendations for each ER Manager based on the trained model.
   - Top-K Recommendations: Retrieves the top `K` branches based on predicted probabilities.
   - Validation: Ensures that recommendations are relevant and accurately reflect historical visit data.

4. Post-Processing:
   - Region Analysis: Identifies the most frequent region from the recommendations for each ER Manager to refine the recommendations.
   - Final Filtering: Ensures that recommendations match the most common region assigned to the ER Manager.

5. Output:
   - Results: Consolidates the recommendations into a final DataFrame, which is filtered to ensure the relevance and accuracy of the suggestions.

6. Validation and Final Checks:
   - Validation Checking (`validation_checking`): Verifies the validity of recommendations against the original data to ensure that branches are allocated correctly.
   - Region Finalization: Assigns the most frequent region to each ER Manager and filters recommendations accordingly.

#### Code Flow:

1. Loading Data:
   - Imports data from various sources and prepares it for training and testing.

2. Model Training:
   - Trains a Random Forest model and tunes hyperparameters to optimize performance.

3. Generating Recommendations:
   - Uses the trained model to predict and generate top branch recommendations for each ER Manager.

4. Processing Results:
   - Creates DataFrames for recommendations, handles missing values, and ensures data consistency.

5. Validation and Filtering:
   - Validates recommendations against historical data and filters by the most common region to ensure relevance.

6. Final Output:
   - Merges and filters results to produce a final recommendation list that is ready for use by ER Managers.

# Summary:
This project involves creating a recommendation system to suggest top branches (SolIDs) to ER Managers based on historical data and model predictions. It includes data preparation, model training and tuning, recommendation generation, validation, and final filtering to ensure that the suggestions are accurate and relevant to each ER Manager.
