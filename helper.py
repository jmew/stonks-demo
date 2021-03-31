import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import tensorboard
# import torch.profiler


def split_data(stock_val, lookback):
    data_raw = stock_val.values # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    
    return [x_train, y_train, x_test, y_test]

def train(num_epochs, model, x_train, y_train, writer, title):
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()

    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        writer.add_scalar(title, loss.item(), t)
        # p.step()
        
    training_time = time.time()-start_time

def predict(model, x_test, y_test, scaler):
    model.eval()
    y_test_pred = model(x_test)

    # invert predictions
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())
    return y_test, y_test_pred

def plot_results(y_test, y_test_pred, df):
    loc = plticker.MultipleLocator(base=25) 

    figure, axes = plt.subplots(figsize=(16, 7))

    dates = df[['Date']][len(df)-len(y_test):].to_numpy().reshape(-1)

    axes.plot(dates, y_test, color = 'red', label = 'Real MSFT Stock Price')
    axes.plot(dates, y_test_pred, color = 'blue', label = 'Predicted MSFT Stock Price')
    axes.xaxis.set_major_locator(loc)

    plt.title('MSFT Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('MSFT Stock Price')
    plt.legend()
    return plt, figure
