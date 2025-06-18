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



train_split = int(len(training) * .85)
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



model = LSTM(input_size=1,hidden_size=512,num_layers=1,output_size=1)
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


### Predicted Vs Actual Federal Fund's Rates

```text
         Date  Actual Interest Rate  Predicted Interest Rate
0  2014-10-01                  0.09                 0.106762
1  2014-11-01                  0.09                 0.106762
2  2014-12-01                  0.12                 0.106762
3  2015-01-01                  0.11                 0.136875
4  2015-02-01                  0.11                 0.126837
5  2015-03-01                  0.11                 0.126837
6  2015-04-01                  0.12                 0.126837
7  2015-05-01                  0.12                 0.136875
8  2015-06-01                  0.13                 0.136875
9  2015-07-01                  0.13                 0.146912
10 2015-08-01                  0.14                 0.146912
11 2015-09-01                  0.14                 0.156949
12 2015-10-01                  0.12                 0.156949
13 2015-11-01                  0.12                 0.136875
14 2015-12-01                  0.24                 0.136875
15 2016-01-01                  0.34                 0.257310
16 2016-02-01                  0.38                 0.357655
17 2016-03-01                  0.36                 0.397789
18 2016-04-01                  0.37                 0.377722
19 2016-05-01                  0.37                 0.387756
          Date  Actual Interest Rate  Predicted Interest Rate
106 2023-08-01                  5.33                 5.132757
107 2023-09-01                  5.33                 5.341501
108 2023-10-01                  5.33                 5.341501
109 2023-11-01                  5.33                 5.341501
110 2023-12-01                  5.33                 5.341501
111 2024-01-01                  5.33                 5.341501
112 2024-02-01                  5.33                 5.341501
113 2024-03-01                  5.33                 5.341501
114 2024-04-01                  5.33                 5.341501
115 2024-05-01                  5.33                 5.341501
116 2024-06-01                  5.33                 5.341501
117 2024-07-01                  5.33                 5.341501
118 2024-08-01                  5.33                 5.341501
119 2024-09-01                  5.13                 5.341501
120 2024-10-01                  4.83                 5.142699
121 2024-11-01                  4.64                 4.844336
122 2024-12-01                  4.48                 4.655277
123 2025-01-01                  4.33                 4.496009
124 2025-02-01                  4.33                 4.346649
125 2025-03-01                  4.33                 4.3466493
```

![predicted_actual](images\predicted-vs-actual.png)



