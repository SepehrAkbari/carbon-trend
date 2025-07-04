import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from preprocess import preprocess

warnings.filterwarnings("ignore")

df = pd.read_csv('../data/annual_global_CO2_levels', 
                 delimiter = "\t", 
                 names = ["Year", "CO2 Level"], 
                 skiprows = 1)

x_train_val, x_test, y_train_val, y_test, x_train, x_val, y_train, y_val = preprocess(df)

def sqrtModel(a: float, b: float, c:float, d:float, x: np.ndarray) -> np.ndarray:
    return a * np.sqrt(b * x + c) + d

a = 0.95
b = 3.5
c = 0.4
d = -0.95

def sqrtError(a: float, b: float, c:float, d:float, x:np.ndarray, y:np.ndarray) -> float:
    return np.mean((np.abs(sqrtModel(a, b, c, d, x) - y)))

learning_rate = 1E-4
epochs = 400

training_loss = []
validation_loss = []

for epoch in range(epochs):
    y_pred_train = sqrtModel(a, b, c, d, x_train)
    
    da = np.mean(np.sign(y_pred_train - y_train) * np.sqrt(b * x_train + c))
    db = np.mean(np.sign(y_pred_train - y_train) * a * x_train / (2 * np.sqrt(b * x_train + c)))
    dc = np.mean(np.sign(y_pred_train - y_train) * a / (2 * np.sqrt(b * x_train + c)))
    dd = np.mean(np.sign(y_pred_train - y_train))

    a -= learning_rate * da
    b -= learning_rate * db
    c -= learning_rate * dc
    d -= learning_rate * dd
    
    loss = sqrtError(a, b, c, d, x_train, y_train)
    training_loss.append(loss)
    
    y_pred_val = sqrtModel(a, b, c, d, x_val)
    val_loss = sqrtError(a, b, c, d, x_val, y_val)
    validation_loss.append(val_loss)
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Training Loss = {loss:.4f}, Validation Loss = {val_loss:.4f}')
        
print("---------------------------------------")
print(f'Final Training Loss: {sqrtError(a, b, c, d, x_train, y_train):.4f}')
print(f'Final Validation Loss: {sqrtError(a, b, c, d, x_val, y_val):.4f}')
print(f'Optimized Parameters: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, d = {d:.4f}')

plt.figure(figsize=(14, 8))
plt.plot(training_loss, label='Training Loss (MAE)', linewidth=3, color='#1b9e77')
plt.plot(validation_loss, label='Validation Loss (MAE)', linewidth=3, color='#d95f02')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.title('Convergence Analysis: Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

x = np.linspace(0, 1, 100)
y = sqrtModel(a, b, c, d, x)

y_pred_test = sqrtModel(a, b, c, d, x_test)
test_loss = sqrtError(a, b, c, d, x_test, y_test)

print(f"Final Test Loss (MAE): {test_loss:.4f}")

plt.figure(figsize=(14, 8))
plt.plot(x, y, linewidth=2, color='#d95f02', label='Sin Model')
plt.scatter(x_train_val, y_train_val, color='#1b9e77', label='Training Data')
plt.scatter(x_test, y_test, color='#e7298a', label='Testing Data')
plt.scatter(x_test, y_pred_test, color='#7570b3', label='Predicted Test Data')
plt.xlabel('Scaled Year')
plt.ylabel('Scaled CO2 Level')
plt.title('SQRT Model vs. Testing Data')
plt.legend()
plt.grid(True)
plt.show()