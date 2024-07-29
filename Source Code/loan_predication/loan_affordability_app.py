import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create the loan affordability prediction function
def predict_loan_affordability(income, expenses):
    # Load your dataset or use a sample dataset
    data = pd.read_csv("loan_data.csv")  # Replace with your dataset

    # Split the data into features (X) and labels (y)
    X = data.drop("loan_status", axis=1)  # Adjust based on your dataset
    y = data["loan_status"]  # Adjust based on your dataset

    # Load your trained model or train a new one
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Make prediction
    prediction = model.predict([[income, expenses]])

    return prediction[0]

# Set up the Streamlit app
def main():
    # Set the app title and description
    st.title("Loan Affordability App")
    st.markdown("Enter your income and expenses to predict loan affordability.")

    # Create the input fields for user data
    income = st.number_input("Income", min_value=0)
    expenses = st.number_input("Expenses", min_value=0)

    # Add a button to trigger the affordability prediction
    if st.button("Calculate"):
        # Perform the affordability prediction
        prediction = predict_loan_affordability(income, expenses)

        # Display the prediction result
        st.markdown("### Affordability Prediction")
        if prediction == 1:
            st.success("You are likely to afford the loan.")
        else:
            st.error("You may have difficulty affording the loan.")

# Run the app
if __name__ == "__main__":
    main()