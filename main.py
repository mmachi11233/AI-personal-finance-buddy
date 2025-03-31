import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import opena


openai.api_key = 'your-api-key'


def collect_user_data():
    print("Enter your last 6 months' expenses:")
    expenses = []
    descriptions = []
    categories = []
    for i in range(6):
        expense = float(input(f"Month {i + 1} expense: "))
        description = input(f"Describe this expense for Month {i + 1}: ")
        category = input(f"Enter category for Month {i + 1} (e.g., Rent, Food, Transport, Entertainment): ")
        expenses.append(expense)
        descriptions.append(description)
        categories.append(category)
    return np.array(expenses).reshape(-1, 1), descriptions, categories


def train_model(expenses):
    months = np.array(range(1, len(expenses) + 1)).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(months, expenses, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_expense(model):
    next_month = np.array([[7]])  # Predicting for month 7
    predicted_expense = model.predict(next_month)[0][0]
    return predicted_expense


def generate_spending_insights(expenses, descriptions, categories):
    avg_spending = np.mean(expenses)
    highest_spending = np.max(expenses)
    lowest_spending = np.min(expenses)
    insights = f"Your average monthly spending is ${avg_spending:.2f}. Your highest spending was ${highest_spending:.2f}, and your lowest spending was ${lowest_spending:.2f}."
    category_spending = pd.DataFrame({'Category': categories, 'Expense': expenses.flatten()}).groupby('Category').sum()
    return insights, category_spending


def generate_budget_suggestions(expenses, predicted_expense):
    avg_spending = np.mean(expenses)
    budget_suggestion = f"Based on your spending trends, consider setting a budget of ${avg_spending * 0.9:.2f} to save more. Try reducing discretionary expenses."
    return budget_suggestion


def ai_chatbot():
    print("Hello! I'm your AI-powered financial advisor. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("AI: Goodbye! Stay financially smart!")
            break
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )
        print(f"AI: {response['choices'][0]['message']['content']}")


def plot_forecast(expenses, model):
    months = np.array(range(1, len(expenses) + 1)).reshape(-1, 1)
    future_months = np.array(range(1, 9)).reshape(-1, 1)  # Forecasting for 2 extra months

    plt.figure(figsize=(8, 5))
    plt.scatter(months, expenses, color='blue', label='Actual Expenses')
    plt.plot(future_months, model.predict(future_months), color='red', linestyle='dashed', label='Forecasted Expenses')
    plt.xlabel('Month')
    plt.ylabel('Expense ($)')
    plt.title('Expense Forecast')
    plt.legend()
    plt.show()



expenses, descriptions, categories = collect_user_data()


model = train_model(expenses)

predicted_expense = predict_expense(model)
print(f"Predicted expense for next month: ${predicted_expense:.2f}")


insights, category_spending = generate_spending_insights(expenses, descriptions, categories)
budget_suggestions = generate_budget_suggestions(expenses, predicted_expense)
print(insights)
print("Category-wise spending:")
print(category_spending)
print(budget_suggestions)


plot_forecast(expenses, model)
   
ai_chatbot()
