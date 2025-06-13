import pandas as pd
import matplotlib
      # if you're using Tkinter backend
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()


# Normalize the features (Z-score normalization)
df['Size'] = (df['Size'] - df['Size'].mean()) / df['Size'].std()
df['Price'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()

# Loss function (Mean Squared Error)
def lossfunction(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Size  
        y = points.iloc[i].Price
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

# Gradient Descent step
def gradient(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))

    for i in range(len(points)):
        x = points.iloc[i].Size
        y = points.iloc[i].Price
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return m, b

# Initialize parameters
m = 0
b = 0
epochs = 1000
L = 0.005  # Learning rate

# Training loop
for i in range(epochs):
    m, b = gradient(m, b, df, L)
    if i % 50 == 0:
        print(f"Epoch {i} - Loss: {lossfunction(m, b, df):.4f}")

# Final parameters
print(f"\nFinal model: y = {m:.4f}x + {b:.4f}")

# Plotting
plt.scatter(df.Size, df.Price, label='Data Points')
x_vals = list(df['Size'])
x_vals.sort()
plt.plot(x_vals, [m * x + b for x in x_vals], color="red", label='Regression Line')
plt.xlabel("Size (Normalized)")
plt.ylabel("Price (Normalized)")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.savefig("regression_plot.png")
plt.show()





