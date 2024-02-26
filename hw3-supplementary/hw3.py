from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    data = np.load(filename)
    centered_data = data - np.mean(data, axis=0)
    return centered_data

def get_covariance(dataset):
    dot_product = np.dot(np.transpose(dataset), dataset)
    covariance_matrix = dot_product * (1 / (len(dataset) - 1))
    return covariance_matrix

def get_eig(S, m):
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[1024-m, 1023])
    eigenvalues = -np.sort(-eigenvalues)
    eigenvectors = np.fliplr(eigenvectors)
    eigenvalues_matrix = np.zeros((m, m))
    np.fill_diagonal(eigenvalues_matrix, eigenvalues)
    return eigenvalues_matrix, eigenvectors

def get_eig_prop(S, prop):
    sum = 0
    index = 1025
    eigenvalues = eigh(S,eigvals_only=True)
    for x in eigenvalues:
        sum += x
    for m in eigenvalues:
        index -= 1
        if (m/sum) > prop:
            break
    return get_eig(S, index)


def project_image(image, U):
    image = np.array(image)
    alpha = np.dot(U.T, image)
    proj = np.dot(U, alpha)
    return proj


def display_image(orig, proj):
    orig_reshaped = np.reshape(orig, (32, 32))
    proj_reshaped = np.reshape(proj, (32, 32))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    
    ax1.set_title("Original")
    ax2.set_title("Projection")
    
    im1 = ax1.imshow(orig_reshaped, aspect='equal', cmap='gray')
    im2 = ax2.imshow(proj_reshaped, aspect='equal', cmap='gray')
    
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    return fig, ax1, ax2
