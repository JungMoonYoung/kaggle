import pandas as pd
import joblib


test = pd.read_csv("../data/preprocessed_test.csv")


rf_model = joblib.load("../models/random_forest_model.pkl")

predictions = rf_model.predict(test)


submission = pd.DataFrame({'id': test['id'], 'sales': predictions})
submission.to_csv("../data/submission.csv", index=False)
print("Predictions saved to '../data/submission.csv'")
