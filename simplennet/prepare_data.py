from sklearn.preprocessing import MinMaxScaler
import numpy as np

input_train = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0], [10, 0, 0], [10, 1, 1], [10, 0, 1]])
output_train = np.array([[0], [0], [0], [1], [1], [1]])
input_pred = np.array([1, 1, 0])

input_test = np.array([[1, 1, 1], [10, 0, 1], [0, 1, 10], [10, 1, 10], [0, 0, 0], [0, 1, 1]])
output_test = np.array([[0], [1], [0], [1], [0], [0]])

scaler = MinMaxScaler()
input_train_scaled = scaler.fit_transform(input_train)
output_train_scaled = scaler.fit_transform(output_train)
input_test_scaled = scaler.fit_transform(input_test)
output_test_scaled = scaler.fit_transform(output_test)

np.save('data/input_train_scaled.npy', input_train_scaled)
np.save('data/output_train_scaled.npy', output_train_scaled)
np.save('data/input_test_scaled.npy', input_test_scaled)
np.save('data/output_test_scaled.npy', output_test_scaled)
np.save('data/input_pred.npy', input_pred)