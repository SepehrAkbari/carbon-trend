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

def linearModel(m: float, b: float, x:np.ndarray) -> float:
    return m * x + b

m = 1.0
b = 0.0

def linearError(m: float, b: float, x:np.ndarray, y:np.ndarray) -> float:
    return np.mean((np.abs(linearModel(m, b, x) - y)))

learning_rate = 1E-3
epochs = 50

training_loss = []
validation_loss = []

for epoch in range(epochs):
    y_pred_train = linearModel(m, b, x_train)
    
    dm = np.mean(np.sign(y_pred_train - y_train) * x_train)
    db = np.mean(np.sign(y_pred_train - y_train))
    
    m -= learning_rate * dm
    b -= learning_rate * db
    
    loss = linearError(m, b, x_train, y_train)
    training_loss.append(loss)
    
    y_pred_val = linearModel(m, b, x_val)
    val_loss = linearError(m, b, x_val, y_val)
    validation_loss.append(val_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Training Loss = {loss:.4f}, Validation Loss = {val_loss:.4f}')
        
print("---------------------------------------")
print(f'Final Training Loss: {linearError(m, b, x_train, y_train):.4f}')
print(f'Final Validation Loss: {linearError(m, b, x_val, y_val):.4f}')
print(f'Optimized Parameters: m = {m:.4f}, b = {b:.4f}')

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
y = linearModel(m, b, x)

y_pred_test = linearModel(m, b, x_test)
test_loss = linearError(m, b, x_test, y_test)

print(f"Final Test Loss (MAE): {test_loss:.4f}")

plt.figure(figsize=(14, 8))
plt.plot(x, y, linewidth=2, color='#d95f02', label='Linear Model')
plt.scatter(x_train_val, y_train_val, color='#1b9e77', label='Training Data')
plt.scatter(x_test, y_test, color='#e7298a', label='Testing Data')
plt.scatter(x_test, y_pred_test, color='#7570b3', label='Predicted Test Data')
plt.xlabel('Scaled Year')
plt.ylabel('Scaled CO2 Level')
plt.title('Linear Model vs. Testing Data')
plt.legend()
plt.grid(True)
plt.show()