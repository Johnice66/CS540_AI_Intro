import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def visualize_data(X, Y):
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, marker='o', linestyle='-', color='b')
    plt.title('Year vs. Number of Frozen Days')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig('plot.jpg')
    plt.show()


def linear_regression(X, Y):
    X_matrix = np.vstack((np.ones_like(X), X)).T
    print("Q3a:")
    print(X_matrix)
    
    Y_vector = Y
    print("Q3b:")
    print(Y_vector)
    
    Z = np.dot(X_matrix.T, X_matrix)
    print("Q3c:")
    print(Z)
    
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)
    
    PI = np.dot(I, X_matrix.T)
    print("Q3e:")
    print(PI)
    
    beta = np.dot(PI, Y_vector)
    print("Q3f:")
    print(beta)

    return beta


def predict(beta, x):
    y_hat = beta[0] + beta[1] * x
    print(f"Q4: Prediction for {x} is {y_hat:.2f} days")
    return y_hat


def interpret_model(beta):
    # Q5a: Determine the sign of beta_1
    sign = "=" if beta[1] == 0 else ("<" if beta[1] < 0 else ">")
    print(f"Q5a: {sign}")
    
    # Q5b: Interpret the sign
    interpretation = (
        "If beta_1 is positive (>), it indicates that as the year increases, "
        "the number of ice days increases, suggesting a trend towards longer "
        "winters over time. If beta_1 is negative (<), it indicates a trend "
        "towards shorter winters over time. If beta_1 is zero (=), it suggests "
        "that the number of ice days does not change significantly with the year."
    )
    print(f"Q5b: {interpretation}")


def model_limitation(beta):
    # Q6a: Predict the year when Lake Mendota will no longer freeze
    x_star = -beta[0] / beta[1]
    print(f"Q6a: {x_star:.2f}")
    
    # Q6b: Discuss the reasonability of the prediction
    discussion = (
        "The prediction that Lake Mendota will no longer freeze in the year "
        f"{x_star:.0f} might not be compelling due to the inherent limitations "
        "of linear regression models, which may not capture the complexity of "
        "climate patterns. Moreover, the model does not consider potential "
        "future interventions or changes in global climate patterns. It's "
        "crucial to approach such predictions with skepticism and consider a "
        "multitude of factors and models."
    )
    print(f"Q6b: {discussion}")


def main(filename):

    data = pd.read_csv(filename)
    
    X = data['year'].values
    Y = data['days'].values
    
    visualize_data(X, Y)
    
    linear_regression(X, Y)

    # Make a prediction about 2022-2023
    beta = linear_regression(X, Y)
    predict(beta, 2022)
    interpret_model(beta)
    model_limitation(beta)



if __name__ == "__main__":
    filename = sys.argv[1]  
    main(filename)
