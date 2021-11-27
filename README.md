# Credit_Risk_Analysis
## Project.
- The purpose of the project is to evaluate credit risk using machine learning models. Using 6 different methods, Naive Random Oversampling, SMOTE Oversampling, Cluster Centroid Undersampling, SMOTEENN Sampling, Balanced Random Forest Classifying and Easy Ensemble Classifying, we are able to determine which method is the most effective at predicting credit risk.

## Results.
1. Naive Random Oversampling.
![Over](https://user-images.githubusercontent.com/85714314/143680617-c03fe802-8b78-47a7-a013-ceb357dc2d9b.png)

- Accuracy Score:65%
- Precision High Risk: 0.01
- Recall High Risk: 0.72

2. SMOTE Oversampling.
![smote](https://user-images.githubusercontent.com/85714314/143680651-45958b2b-48ca-4bbb-b223-d99461484f18.png)

- Accuracy Score: 61%
- Precision High Risk: 0.01
- Recall High Risk: 0.54

3. Cluster Centroid Undersampling.
![under](https://user-images.githubusercontent.com/85714314/143680675-f242105d-945a-456f-ac40-db2935e3ff63.png)

- Accuracy Score: 58%
- Precision High Risk: 0.01
- Recall High Risk: 0.53

4. SMOTEENN Sampling.
![combo](https://user-images.githubusercontent.com/85714314/143680845-a047b08b-3eb7-4f0c-8ddd-1e062c5cb33e.png)

- Accuracy Score: 65%
- Precision High Risk: 0.01
- Recall High Risk: 0.72

5. Balanced Forest Classifying.
![RFC](https://user-images.githubusercontent.com/85714314/143680781-1c560039-61e9-435a-8c46-ca5861a46002.png)

- Accuracy Score: 66%
- Precision High Risk: 0.69
- Recall High Risk: 0.33

6. Easy Ensemble Classifying.
![eec](https://user-images.githubusercontent.com/85714314/143681760-83ec83f8-292f-45a1-96e3-f77e84e0a02f.png)

- Accuracy Score: 82%
- Precision High Risk: 0.03
- Recall High Risk: 0.76

## Summary
- The analysis is looking to find which model is most effective for predicting credit risk. Looking at the recall rate for high risk, the model that scored that highest was Easy Ensemble Classifying. In conjuction with also having the highest accuracy rate, this model is the most effective for determining if a loan is high risk or not.



