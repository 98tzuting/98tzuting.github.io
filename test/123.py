# date:  9.20

import numpy as np
import matplotlib.pyplot as plt

def calculate_E(k, a, ts0, t, matrix_size):
    eigenvalue_results = []
    eigenvector_results = []   # Store eigenvectors

    for i in range(len(k)):
        M = np.zeros((matrix_size, matrix_size), dtype=complex)
        for column in range(matrix_size):
            for row in range(matrix_size):

                # First matrix conditions
                if column %2 == 0 and (row - column) == 1:
                    M[column, row] += -t * np.exp(-1j * k[i] * a) - t
                elif (column) %2 == 1 and (column - row) == 1:
                    M[column, row] += -t * np.exp(1j * k[i] * a) - t
                elif (row%4==1) and (column - row)==2 :
                    M[column, row] += -t
                elif (column%4==0) and column!=0 and (column - row)==2 :
                    M[column, row] += -t
                elif (column%4==1) and (row - column)==2 :
                   M[column, row] += -t
                elif (row%4==0) and row!=0 and (row - column)==2 :
                    M[column, row] += -t

            # Second matrix conditions
                if (column == row) and ((column-3)%4==0 or column%4==0):
                   M[column, row] += -1j*ts0*np.exp(-1j * k[i] * a) + 1j*ts0*np.exp(1j*k[i]*a)
                elif (column == row) and ((column-1)%4==0 or (column-2)%4==0):
                   M[column, row] += -1j*ts0*np.exp(1j * k[i] * a) + 1j*ts0*np.exp(-1j*k[i]*a)
                elif ((column-1)%4==0 and (row-2)%4 == 0 and abs(column-row)==1) or ((column-5)%4==0 and (row-2)%4==0 and abs(column-row)==3):
                   M[column, row] += 1j*ts0*np.exp(1j*k[i]*a) + (-1j)*ts0
                elif ((column-3)%4==0 and (row-4)%4 == 0 and abs(column-row)==3) or ((column-3)%4==0 and row%4==0 and abs(column-row)==1):
                    M[column, row] += 1j*ts0 + (-1j)*ts0*np.exp(1j*k[i]*a)
                elif ((column-2)%4==0 and (row-1)%4==0 and abs(column-row)==1) or ((column-2)%4==0 and (row-5)%4==0 and abs(column-row)==3) :
                    M[column, row] += 1j*ts0 + (-1j)*ts0*np.exp(-1j*k[i]*a)
                elif (column%4==0 and (row-3)%4==0 and abs(column-row)==3) or ((column-4)%4==0 and (row-3)%4==0 and abs(column-row)==1):
                    M[column, row] += (-1j)*ts0 + (1j)*ts0*np.exp(-1j*k[i]*a)

        eigenvalues, eigenvectors = np.linalg.eigh(M) # Get both eigenvalues and eigenvectors
        eigenvalue_results.append(eigenvalues)
        eigenvector_results.append(eigenvectors)

    eigenvalue_results = np.array(eigenvalue_results)
    eigenvector_results = np.array(eigenvector_results)

    return eigenvalue_results, eigenvector_results


# Parameters
t = 1
ts0 = 0.1
a = 1
matrix_size = 8


def f_func(Ek, w):  # Change eps to w
    gamma = 1e-4
    return (1/(2*np.pi*np.pi)) * (gamma / ((w-Ek)**2 + gamma**2))

def compute_total_D(w_values, matrix_size, k_values):
    total_D_values = np.zeros(len(w_values))
    delta_k = k_values[1] - k_values[0]
    E_values = [calculate_E([ki], a, ts0, t, matrix_size)[0][0] for ki in k_values]
    E_derivatives = np.gradient(E_values, delta_k, axis=0)  # gradient along the k-values
    print(delta_k)
    # print(E_derivatives)
    # For each w value
    for j, w in enumerate(w_values):
        integral_sum = 0.0

        # For each k value and its corresponding energy bands and derivatives
        for ki, E, dE in zip(k_values, E_values, E_derivatives):
            mask = E > 0  # considering positive energy values
            E_positive = E[mask]
            dE_positive = dE[mask]

            for e, de in zip(E_positive, dE_positive):
                integral_sum += f_func(e, w) / np.abs(de) * delta_k  # delta_k added for integration step
        
        total_D_values[j] = integral_sum

    return total_D_values





k_values = np.linspace(-2*np.pi, 2*np.pi, 1000)  # More points for better integration
eps_values = np.linspace(0, 5, 300)
D_results = compute_total_D(eps_values, matrix_size, k_values)

# Plotting the density of states D(Epsilon)
plt.figure(figsize=(10,6))
plt.plot(eps_values, D_results)
plt.xlabel('Epsilon')
plt.ylabel('Total D(Epsilon)')
plt.title('Total Density of states D(Epsilon) vs Epsilon')
plt.grid(True)
# plt.show()
plt.savefig("123.png")


# Plotting the bands E(k) using conditions
bands = []
for ki in k_values:  # Use k_values here
    E = calculate_E([ki], a, ts0, t, matrix_size)
    mask = E[0] > 0
    E_positive = E[0][mask]
    bands.append(E_positive)
print(ki)
# Fixing the plotting of bands to handle variable lengths of positive bands
plt.figure(figsize=(10,6))
max_band_length = max(map(len, bands))
for i in range(max_band_length):
    plt.plot([band[i] if i < len(band) else np.nan for band in bands], k_values)

plt.xlabel('E(k)')
plt.ylabel('k')
plt.title('Filtered bands E(k) vs k')
plt.grid(True)
# plt.show()
plt.savefig("1234.png")

print("finished")