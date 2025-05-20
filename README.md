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

### Structure
This will be setup pretty much the same as most timeseries LSTM's via torch.
```python
training = data.iloc[:,0:1].values


train_split = int(len(training) * .70)

train_data = training[:train_split]
test_data = training[train_split:]
train_data.shape
test_data.shape

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def slider(df, seq_length):
    X, y = [], []
    for i in range(len(df) - seq_length):
        X_ = df[i:(i + seq_length)]
        y_ = df[i + seq_length]
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
```
I do not simply use "seq_length = 1" every single time I forecast. The only reason is that the train size for the split was very good to run 1000 epochs for the lstm and I did not see any harm

### results after training
```python
with torch.no_grad():
    pred = model(X_test)
    pred_np = pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    pred_rescaled = scaler.inverse_transform(pred_np)
    actual_rescaled = scaler.inverse_transform(y_test_np)


test_dates = data.iloc[train_split + seq_length:]['Date'].reset_index(drop=True)

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
234 2023-09-01                  5.33                 5.349109
235 2023-10-01                  5.33                 5.349109
236 2023-11-01                  5.33                 5.349109
237 2023-12-01                  5.33                 5.349109
238 2024-01-01                  5.33                 5.349109
239 2024-02-01                  5.33                 5.349109
240 2024-03-01                  5.33                 5.349109
241 2024-04-01                  5.33                 5.349109
242 2024-05-01                  5.33                 5.349109
243 2024-06-01                  5.33                 5.349109
244 2024-07-01                  5.33                 5.349109
245 2024-08-01                  5.33                 5.349109
246 2024-09-01                  5.13                 5.349109
247 2024-10-01                  4.83                 5.150422
248 2024-11-01                  4.64                 4.852217
249 2024-12-01                  4.48                 4.663247
250 2025-01-01                  4.33                 4.504051
251 2025-02-01                  4.33                 4.354753
252 2025-03-01                  4.33                 4.354753
253 2025-04-01                  4.33                 4.354753
```

![predicted_actual](images/predicted_actual_interestrates.png)


