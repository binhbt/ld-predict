# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from xls_util import load_data
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq = load_data('LOTO DATA.xlsx', 0)
for i in range(1, 16):
	print(i)
	data = load_data('LOTO DATA.xlsx', i)
	raw_seq.extend(data)
# raw_seq =array([55, 12, 53, 36, 70, 51, 70, 7, 75, 13, 99, 54, 94, 97, 25, 80, 63, 92, 14, 92, 5, 23, 75, 31, 92, 22, 52, 59, 34, 41, 42, 9, 0, 56, 14, 24, 84, 0, 65, 99, 49, 44, 50, 88, 6, 70, 15, 3, 75, 41, 27, 9, 30, 21, 44, 26, 55, 28, 10, 64, 40, 96, 97, 57, 30, 10, 21, 61, 89, 96, 80, 55, 49, 16, 45, 21, 39, 20, 37, 84, 33, 48, 33, 87, 20, 33, 2, 37, 73, 79, 77, 92, 98, 73, 23, 72, 46, 19, 30, 15, 78, 95, 6, 58, 78, 86, 6, 79, 0, 69, 73, 22, 17, 1, 16, 13, 58, 76, 35, 16, 86, 34, 19, 13, 28, 13, 79, 37, 30, 4, 46, 47, 27, 39, 20, 48, 78, 79, 87, 86, 2, 36, 31, 85, 64, 12, 46, 49, 75, 43, 44, 7, 40, 66, 80, 73, 22, 57, 95, 29, 10, 49, 49, 20, 21, 82, 73, 3, 17, 28, 83, 67, 28, 78, 91, 98, 2, 50, 10, 10, 38, 97, 33, 9, 44, 14, 45, 78, 83, 65, 45, 79, 15, 21, 64, 73, 31, 57, 40, 81, 94, 98, 47, 39, 54, 61, 99, 16, 1, 31, 88, 55, 43, 11, 40, 66, 98, 48, 26, 14, 39, 98, 20, 73, 80, 17, 24, 55, 72, 17, 7, 79, 43, 41, 57, 45, 34, 1, 7, 9, 4, 91, 98, 53, 27, 14, 2, 59, 7, 97, 81, 63, 6, 37, 83, 10, 41, 11, 14, 83, 73, 46, 16, 6, 78, 91, 56, 70, 82, 91, 77, 74, 58, 37, 89, 4, 74, 22, 54, 78, 2, 7, 37, 34, 6, 94, 48, 12, 60, 47, 82, 75])

# choose a number of time steps
n_steps = 36
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# model.fit(X, y, epochs=200, verbose=0)
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (n_steps, 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X, y, epochs = 100, batch_size = 32)
# demonstrate prediction
# raw_seq2 = load_data('LOTO DATA.xlsx', 17)

x_input = array([13, 5, 50, 27, 63, 49, 84, 57, 60, 84, 2, 0, 52, 16, 32, 45, 27, 17, 35, 11, 56, 90, 96, 79, 51, 2, 97, 96, 66, 4, 14, 43, 32, 39, 40, 15])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = regressor.predict(x_input, verbose=0)
print(yhat)