### Forecasting Federal Fund's Rate

I used the Fred Api token to gain access to to the series data titled "FEDFUNDS". This is a pretty similar forecast compared to all of the other LSTM, torch forecasts. You need the Api key from Fred to gain access to their data. This is a forecast of the Federal Funds Rate.

### Requirements
```bash
pip install fredapi torch torchvision torchaudio numpy pandas sciki-learn matplotlib seaborn
```
Make Sure, if you ever use Fred for data, to hide your Api key from the public. Put it in a .gitignore file or something else. Make sure it is not publicly available.

### Accessing Data from Fred
When you get access to your Api key, make a .env file or a package to hide the token id.
```python

fred = fr.Fred(fred_api_key)



device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

"""Interest Rates(Fed Funds Rate) """

ffr = fred.get_series("FEDFUNDS")
ffr.name = "Fed Funds Rate"

data = pd.DataFrame(ffr).dropna()
```
When you name series data, you first have to use your Api via "fredapi" in python(if you are using python). Then, put it in a dataframe and the rest goes as usual.


![federal_funds_rate_plot](images/federal-funds-rate-date.png)



### Structure
This will be setup pretty much the same as most timeseries LSTM's via torch.
```python
training = data.iloc[:,0:1].values



train_split = int(len(training) * .70)
train_data = training[:train_split]
test_data = training[train_split:]


scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def slider(dataframe, seq_length):
    X, y = [], []
    for i in range(len(dataframe) - seq_length - 1):
        X_ = dataframe[i:(i + seq_length)]
        y_ = dataframe[i + seq_length]
        X.append(X_)
        y.append(y_)
    return np.array(X), np.array(y)

seq_length = 1
X_train, y_train = slider(train_data, seq_length)
X_test,y_test = slider(test_data,seq_length)


X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()





class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,X):
        h0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        out,_ = self.lstm(X,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out



model = LSTM(input_size=1,hidden_size=64,num_layers=1,output_size=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
loss_fn = nn.MSELoss()
epochs = 500

for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred.float(),y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rsme = np.sqrt(loss_fn(y_pred,y_train))
        y_pred_test =  model(X_test)
        test_rsme = np.sqrt(loss_fn(y_pred_test,y_test))
        print(f'Epoch: {epoch}; train_RSEM: {train_rsme:.4}; Test RSME: {test_rsme:.4}')
```
I do not simply use "seq_length = 1" every single time I forecast. The only reason is that the train size for the split was very good to run 500 epochs for the lstm and I did not see any harm

### results after training
```python
with torch.no_grad():
    pred = model(X_test)
    pred_np = pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    pred_rescaled = scaler.inverse_transform(pred_np)
    actual_rescaled = scaler.inverse_transform(y_test_np)


test_dates = data.iloc[train_split + seq_length:-1]['Date'].reset_index(drop=True)

comparison_df = pd.DataFrame({
    "Date": test_dates,
    "Actual Interest Rate": actual_rescaled.flatten(),
    "Predicted Interest Rate": pred_rescaled.flatten()
})

print(comparison_df.tail(20))
```

### Predicted Vs Actual Federal Fund's Rates

```text
Date  Actual Interest Rate  Predicted Interest Rate
0  2004-03-01                  1.00                 1.255862
1  2004-04-01                  1.00                 1.246838
2  2004-05-01                  1.00                 1.246838
3  2004-06-01                  1.03                 1.246838
4  2004-07-01                  1.26                 1.273913
5  2004-08-01                  1.43                 1.481956
6  2004-09-01                  1.61                 1.636256
7  2004-10-01                  1.76                 1.800118
8  2004-11-01                  1.93                 1.937050
9  2004-12-01                  2.16                 2.092651
10 2005-01-01                  2.28                 2.303862
11 2005-02-01                  2.50                 2.414372
12 2005-03-01                  2.63                 2.617525
13 2005-04-01                  2.79                 2.737904
14 2005-05-01                  3.00                 2.886398
15 2005-06-01                  3.04                 3.081854
16 2005-07-01                  3.26                 3.119155
17 2005-08-01                  3.50                 3.324714
18 2005-09-01                  3.62                 3.549733
19 2005-10-01                  3.78                 3.662541
          Date  Actual Interest Rate  Predicted Interest Rate
234 2023-09-01                  5.33                 5.290770
235 2023-10-01                  5.33                 5.290770
236 2023-11-01                  5.33                 5.290770
237 2023-12-01                  5.33                 5.290770
238 2024-01-01                  5.33                 5.290770
239 2024-02-01                  5.33                 5.290770
240 2024-03-01                  5.33                 5.290770
241 2024-04-01                  5.33                 5.290770
242 2024-05-01                  5.33                 5.290770
243 2024-06-01                  5.33                 5.290770
244 2024-07-01                  5.33                 5.290770
245 2024-08-01                  5.33                 5.290770
246 2024-09-01                  5.13                 5.290770
247 2024-10-01                  4.83                 5.098405
248 2024-11-01                  4.64                 4.810790
249 2024-12-01                  4.48                 4.629220
250 2025-01-01                  4.33                 4.476680
251 2025-02-01                  4.33                 4.333973
252 2025-03-01                  4.33                 4.333973
253 2025-04-01                  4.33                 4.333973
```

![predicted_actual](images\predicted-vs-actual.png)



