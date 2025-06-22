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

data = data.shift(1)
data.dropna(inplace=True)
training = data.iloc[:,0:1].values



train_split = int(len(training) * .88)
train_data = training[:train_split]
test_data = training[train_split:]
print(train_data.shape)
print(test_data.shape)

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



model = LSTM(input_size=1,hidden_size=64,num_layers=2,output_size=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
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
        print(f'Epoch: {epoch}; train_RSEM: {train_rsme:.4}; Test RSME: {test_rsme:.4}')v
```




### Predicted Vs Actual Fed Funds Rate

```text
Date  Actual Interest Rate  Predicted Interest Rate
0  2016-12-01                  0.54                 0.445663
1  2017-01-01                  0.65                 0.572765
2  2017-02-01                  0.66                 0.680426
3  2017-03-01                  0.79                 0.690219
4  2017-04-01                  0.90                 0.817600
5  2017-05-01                  0.91                 0.925493
6  2017-06-01                  1.04                 0.935306
7  2017-07-01                  1.15                 1.062954
8  2017-08-01                  1.16                 1.171068
9  2017-09-01                  1.15                 1.180902
10 2017-10-01                  1.15                 1.171068
11 2017-11-01                  1.16                 1.171068
12 2017-12-01                  1.30                 1.180902
13 2018-01-01                  1.41                 1.318648
14 2018-02-01                  1.42                 1.426982
15 2018-03-01                  1.51                 1.436835
16 2018-04-01                  1.69                 1.525545
17 2018-05-01                  1.70                 1.703138
18 2018-06-01                  1.82                 1.713011
19 2018-07-01                  1.91                 1.831541
         Date  Actual Interest Rate  Predicted Interest Rate
80 2023-08-01                  5.33                 5.119862
81 2023-09-01                  5.33                 5.330294
82 2023-10-01                  5.33                 5.330294
83 2023-11-01                  5.33                 5.330294
84 2023-12-01                  5.33                 5.330294
85 2024-01-01                  5.33                 5.330294
86 2024-02-01                  5.33                 5.330294
87 2024-03-01                  5.33                 5.330294
88 2024-04-01                  5.33                 5.330294
89 2024-05-01                  5.33                 5.330294
90 2024-06-01                  5.33                 5.330294
91 2024-07-01                  5.33                 5.330294
92 2024-08-01                  5.33                 5.330294
93 2024-09-01                  5.13                 5.330294
94 2024-10-01                  4.83                 5.129882
95 2024-11-01                  4.64                 4.829382
96 2024-12-01                  4.48                 4.639156
97 2025-01-01                  4.33                 4.479032
98 2025-02-01                  4.33                 4.328974
99 2025-03-01                  4.33                 4.328974
```

![predicted_actual](images\predicted-vs-actual.png)



