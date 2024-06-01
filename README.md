# Heart Attack Prediction
## Model Evaluation
To enhance your evaluation of the Random Forest Classifier for predicting heart attacks, consider incorporating a broader range of evaluation metrics and visualizations. Specifically, for a classification problem like this, metrics like precision, recall, and the F1-score can provide more insight into the performance of your model, especially when dealing with imbalanced datasets. Additionally, using ROC curves and calculating the Area Under the Curve (AUC) can help visualize and quantify the trade-off between the true positive rate and the false positive rate.

### Calculate Additional Evaluation Metrics
We calculate the ```Precision```, ```Recall```, ```F1-score```, ```ROC-AUC```  metrics to give a more comprehensive evaluation of the classifier's performance. These metrics are particularly useful in the context of imbalanced datasets, where accuracy alone can be misleading.
```python
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC-AUC:", roc_auc)
```

### ROC curve
Plotted the ROC curve to visualize the performance of the classifier across different threshold settings. The ROC-AUC score provides a single number summary of the classifier's performance, with a higher score indicating a better performing model.
```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
```

### Predictions
By using these additional metrics and visualizations, we will get a better understanding of how well our model performs in predicting heart attacks, beyond just the accuracy score. This approach also helps in identifying any potential improvements that can be made in terms of balancing precision and recall, which are critical in medical diagnosis applications.
 - Make predictions and evaluate the model:
```python
y_pred = best_rf.predict(X_testb)
y_pred_proba = best_rf.predict_proba(X_testb)[:, 1]
evaluate_model(y_testb, y_pred, y_pred_proba)
```
 - Sample patient data:
```python
# Predict for the sample patient data
patient_data = pd.DataFrame({
    'Age': [60],
    'Gender': [1],  # 1 for male, 0 for female
    'Heart rate': [78],
    'Systolic blood pressure': [125],
    'Diastolic blood pressure': [87],
    'weight': [40],
    'height': [1.7],
    'BMI': [24.5],
    'pulse_pressure': [40],
    'BP_based_Condition': [0]  # 0 for Normal, 1 for Pre-Hypertension, 2 for Hypertension
})
```
 - Process the data
```python
# Encode the categorical value
patient_data['BP_based_Condition'] = le.transform(patient_data['BP_based_Condition'])

# Make predictions
prediction = best_rf.predict(patient_data)
prediction_proba = best_rf.predict_proba(patient_data)[:, 1]
```
 - Print the prediction result
```python
print(f"Prediction: {'Heart Attack' if prediction[0] == 1 else 'No Heart Attack'}, Probability: {prediction_proba[0]:.2f}")
```
## Model Interpretation
Interpreting the model results and understanding the impact of features on predictions can be achieved through various methods such as feature importance plots, SHAP values, and LIME. These techniques help in making the black-box models more interpretable and provide insights into how each feature contributes to the final prediction.

### Feature Importance Plots:
The feature importance attribute of Random Forest gives a quick overview of which features are contributing the most to the model's decisions.
```python
# Feature importance from Random Forest
importances = best_rf.feature_importances_
feature_names = X_trainb.columns
```
<div align = 'center'>
  <img src = 'https://github.com/AnxiousCodeGeek/heartAttack-modelEvaluation/assets/138652868/3c377402-e33d-40c7-a18a-f70ddebaf0fa' width = 700 height = 400>
</div>

### SHAP Values:
```SHAP``` (SHapley Additive exPlanations) values provide a unified measure of feature importance and can explain individual predictions.
```python
# Initialize the explainer with the trained Random Forest model
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_testa)

# Check if the model is binary or multiclass and handle accordingly
if isinstance(shap_values, list) and len(shap_values) > 1:
    # For multiclass
    for i, shap_value in enumerate(shap_values):
        print(f"Class {i} SHAP summary plot")
        plt.figure(figsize=(15, 8))
        shap.summary_plot(shap_values, X_testa, feature_names=X_testa.columns,show = False)
else:
    # For binary
    plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_values, X_testa, feature_names=X_testa.columns, show = False)
```
**P.S. My code was not plotting the explanation for all features. I wasn't able to solve this issue, if anyone does, please also let me know.**
<div align = 'center'>
  <img src = 'https://github.com/AnxiousCodeGeek/heartAttack-modelEvaluation/assets/138652868/7d998ba6-13a4-4f71-a067-a7346a09fd3b' width = 300 height = 500>
</div>

### LIME:
```LIME``` (Local Interpretable Model-agnostic Explanations) explains the predictions of any classifier by perturbing the input and understanding how the predictions change.

```python
# Initialize the explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_trainb),
    feature_names=feature_names,
    class_names=['No Heart Attack', 'Heart Attack'],
    mode='classification'
)

# Explain a prediction for a sample
exp = explainer.explain_instance(
    data_row=np.array(X_testb.iloc[0]),
    predict_fn=best_rf.predict_proba
)
```
<div align = 'center'>
  <img src = 'https://github.com/AnxiousCodeGeek/heartAttack-modelEvaluation/assets/138652868/f4dc55b0-542b-45b1-89f2-6e6bd5833a4f' width = 900 height = 300>
</div>

## Model Validation
To evaluate model robustness and stability, we used:
1. **Bootstrapping for Model Robustness and Stability:** This technique involves repeatedly sampling the dataset with replacement to create multiple bootstrap samples, training the model on these samples, and evaluating the performance across these samples.
```python
Bootstrap Accuracies: [0.7808564231738035, 0.7884130982367759, 0.7632241813602015, 0.7858942065491183, 0.7657430730478589, 0.7808564231738035, 0.7783375314861462, 0.7783375314861462, 0.7884130982367759, 0.8110831234256927, 0.8035264483627204, 0.783375314861461, 0.801007556675063, 0.7808564231738035, 0.783375314861461, 0.7556675062972292, 0.8060453400503779, 0.7984886649874056, 0.8035264483627204, 0.7884130982367759, 0.7858942065491183, 0.801007556675063, 0.7858942065491183, 0.783375314861461, 0.7984886649874056, 0.7632241813602015, 0.7934508816120907, 0.7909319899244333, 0.7808564231738035, 0.7783375314861462, 0.8110831234256927, 0.7808564231738035, 0.7732997481108312, 0.8060453400503779, 0.760705289672544, 0.7909319899244333, 0.8060453400503779, 0.818639798488665, 0.7732997481108312, 0.7808564231738035, 0.7909319899244333, 0.7682619647355163, 0.7909319899244333, 0.7858942065491183, 0.7959697732997482, 0.7858942065491183, 0.7909319899244333, 0.7682619647355163, 0.783375314861461, 0.8035264483627204, 0.7783375314861462, 0.7682619647355163, 0.7884130982367759, 0.7884130982367759, 0.7758186397984886, 0.7959697732997482, 0.7858942065491183, 0.7732997481108312, 0.801007556675063, 0.7732997481108312, 0.8060453400503779, 0.7783375314861462, 0.7808564231738035, 0.7909319899244333, 0.7858942065491183, 0.7732997481108312, 0.8136020151133502, 0.8110831234256927, 0.783375314861461, 0.7707808564231738, 0.8035264483627204, 0.7884130982367759, 0.783375314861461, 0.783375314861461, 0.7581863979848866, 0.7732997481108312, 0.7959697732997482, 0.7959697732997482, 0.7732997481108312, 0.8035264483627204, 0.7909319899244333, 0.801007556675063, 0.7682619647355163, 0.7959697732997482, 0.7858942065491183, 0.7783375314861462, 0.7858942065491183, 0.7884130982367759, 0.7732997481108312, 0.7783375314861462, 0.7984886649874056, 0.8035264483627204, 0.7531486146095718, 0.7934508816120907, 0.7758186397984886, 0.7657430730478589, 0.760705289672544, 0.801007556675063, 0.8110831234256927, 0.7632241813602015]
Mean Bootstrap Accuracy: 0.7860705289672544
Standard Deviation of Bootstrap Accuracies: 0.014236774927312207
```
2. **Monte Carlo Simulation:** This involves repeatedly splitting the dataset into training and test sets in different ways, training the model, and evaluating performance to assess the variability in model predictions.

```python
Simulation Accuracies: [0.818639798488665, 0.8060453400503779, 0.8060453400503779, 0.801007556675063, 0.8060453400503779, 0.8161209068010076, 0.818639798488665, 0.7959697732997482, 0.8060453400503779, 0.8287153652392947, 0.7934508816120907, 0.801007556675063, 0.8161209068010076, 0.801007556675063, 0.801007556675063, 0.801007556675063, 0.801007556675063, 0.8110831234256927, 0.8085642317380353, 0.8161209068010076, 0.8110831234256927, 0.8085642317380353, 0.8060453400503779, 0.7959697732997482, 0.7909319899244333, 0.8035264483627204, 0.8085642317380353, 0.8161209068010076, 0.8136020151133502, 0.8161209068010076, 0.8035264483627204, 0.8211586901763224, 0.8161209068010076, 0.8060453400503779, 0.8236775818639799, 0.8136020151133502, 0.8136020151133502, 0.8261964735516373, 0.7934508816120907, 0.8035264483627204, 0.8211586901763224, 0.8136020151133502, 0.801007556675063, 0.836272040302267, 0.8110831234256927, 0.8110831234256927, 0.8161209068010076, 0.8211586901763224, 0.8035264483627204, 0.801007556675063, 0.8035264483627204, 0.7959697732997482, 0.818639798488665, 0.818639798488665, 0.801007556675063, 0.7934508816120907, 0.8136020151133502, 0.8161209068010076, 0.7934508816120907, 0.8035264483627204, 0.8060453400503779, 0.8060453400503779, 0.8035264483627204, 0.7959697732997482, 0.8287153652392947, 0.8110831234256927, 0.801007556675063, 0.7884130982367759, 0.8136020151133502, 0.7884130982367759, 0.836272040302267, 0.8161209068010076, 0.8035264483627204, 0.8085642317380353, 0.7984886649874056, 0.818639798488665, 0.818639798488665, 0.7884130982367759, 0.7959697732997482, 0.8236775818639799, 0.8136020151133502, 0.7984886649874056, 0.7934508816120907, 0.7984886649874056, 0.7758186397984886, 0.8136020151133502, 0.801007556675063, 0.818639798488665, 0.7959697732997482, 0.7858942065491183, 0.8035264483627204, 0.8136020151133502, 0.8035264483627204, 0.8136020151133502, 0.8060453400503779, 0.7984886649874056, 0.8236775818639799, 0.801007556675063, 0.8085642317380353, 0.818639798488665]
Mean Simulation Accuracy: 0.8077581863979847
Standard Deviation of Simulation Accuracies: 0.010944171323217868
```
