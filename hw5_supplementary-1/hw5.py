import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def main(filename):

    data = pd.read_csv(filename)

    X = data['year'].values
    Y = data['days'].values

    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, marker='o', linestyle='-', color='b')
    plt.title('Year vs. Number of Frozen Days')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig('plot.jpg')
    plt.show()

    X_matrix = np.vstack((np.ones_like(X), X)).T
    print("Q3a:", end="\r\n")
    print(X_matrix, end="\r\n")
    
    Y_vector = Y
    print("Q3b:", end="\r\n")
    print(Y_vector, end="\r\n")
    
    Z = np.dot(X_matrix.T, X_matrix)
    print("Q3c:", end="\r\n")
    print(Z, end="\r\n")
    
    I = np.linalg.inv(Z)
    print("Q3d:", end="\r\n")
    print(I, end="\r\n")
    
    PI = np.dot(I, X_matrix.T)
    print("Q3e:", end="\r\n")
    print(PI, end="\r\n")
    
    beta = np.dot(PI, Y_vector)
    print("Q3f:", end="\r\n")
    print(beta, end="\r\n")

    y_hat = beta[0] + beta[1] * 2022
    print(f"Q4:{y_hat}", end="\r\n")
    
    sign = "=" if beta[1] == 0 else ("<" if beta[1] < 0 else ">")
    print(f"Q5a: {sign}", end="\r\n")
    
    interpretation = (
        "If beta_1 is positive (>), it indicates that as the year increases, "
        "the number of ice days increases, suggesting a trend towards longer "
        "winters over time. If beta_1 is negative (<), it indicates a trend "
        "towards shorter winters over time. If beta_1 is zero (=), it suggests "
        "that the number of ice days does not change significantly with the year."
    )
    print(f"Q5b: {interpretation}", end="\r\n")

    
    x_star = -beta[0] / beta[1]
    print(f"Q6a: {x_star}", end="\r\n")
    
    discussion = (
        "The prediction that Lake Mendota will no longer freeze in the year "
        f"{x_star:.0f} might not be compelling due to the inherent limitations "
        "of linear regression models, which may not capture the complexity of "
        "climate patterns. Moreover, the model does not consider potential "
        "future interventions or changes in global climate patterns. It's "
        "crucial to approach such predictions with skepticism and consider a "
        "multitude of factors and models."
    )
    print(f"Q6b: {discussion}", end="\r\n")

if __name__ == "__main__":
    filename = sys.argv[1]  
    main(filename)
