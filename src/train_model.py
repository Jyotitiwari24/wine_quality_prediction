from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


def train_model(X, Y, test_size=0.2, random_state=2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/winequality-red.csv')
    X = df.drop(['quality', 'type'], axis=1)  # drop non-numeric
    Y = df['quality'].apply(lambda y: 1 if y >= 7 else 0)
    
    # Train
    model = train_model(X, Y)

