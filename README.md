### Forecasting Federal Fund's Rate

I used the Fred Api token to gain access to to the series data titled "FEDFUNDS". This is a pretty similar forecast compared to all of the other LSTM, torch forecasts. You need the Api key from Fred to gain access to their data. This is a forecast of the Federal Funds Rate.

### Requirements
```bash
pip install fredapi torch torchvision torchaudio numpy pandas sciki-learn matplotlib seaborn catboost xgboot-cpu mlflow
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

seq_length = 7

X_train, y_train = slider(train_data, seq_length)
X_test,y_test = slider(test_data,seq_length)


X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()





class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, X):
        h0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size)
        c0 = torch.zeros(2 * self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.lstm(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



model = BiLSTM(input_size=1,hidden_size=128,num_layers=2,output_size=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.MSELoss()
epochs = 500
```




### Predicted Vs Actual Fed Funds Rate using Bidirectional LSTM

```text
         Date  Actual Interest Rate  Predicted Interest Rate
0  2017-06-01                  1.04                 0.992218
1  2017-07-01                  1.15                 1.114385
2  2017-08-01                  1.16                 1.238771
3  2017-09-01                  1.15                 1.272003
4  2017-10-01                  1.15                 1.261230
5  2017-11-01                  1.16                 1.247898
6  2017-12-01                  1.30                 1.247165
7  2018-01-01                  1.41                 1.365295
8  2018-02-01                  1.42                 1.490438
9  2018-03-01                  1.51                 1.527278
10 2018-04-01                  1.69                 1.600128
11 2018-05-01                  1.70                 1.759459
12 2018-06-01                  1.82                 1.800849
13 2018-07-01                  1.91                 1.900696
14 2018-08-01                  1.91                 1.989218
15 2018-09-01                  1.95                 1.998279
16 2018-10-01                  2.19                 2.021800
17 2018-11-01                  2.20                 2.224164
18 2018-12-01                  2.27                 2.278809
19 2019-01-01                  2.40                 2.339249
         Date  Actual Interest Rate  Predicted Interest Rate
75 2023-09-01                  5.33                 5.313292
76 2023-10-01                  5.33                 5.343698
77 2023-11-01                  5.33                 5.334824
78 2023-12-01                  5.33                 5.318294
79 2024-01-01                  5.33                 5.307239
80 2024-02-01                  5.33                 5.308608
81 2024-03-01                  5.33                 5.314802
82 2024-04-01                  5.33                 5.314802
83 2024-05-01                  5.33                 5.314802
84 2024-06-01                  5.33                 5.314802
85 2024-07-01                  5.33                 5.314802
86 2024-08-01                  5.33                 5.314802
87 2024-09-01                  5.13                 5.314802
88 2024-10-01                  4.83                 5.120794
89 2024-11-01                  4.64                 4.788943
90 2024-12-01                  4.48                 4.549984
91 2025-01-01                  4.33                 4.385576
92 2025-02-01                  4.33                 4.253560
93 2025-03-01                  4.33                 4.260665
94 2025-04-01                  4.33                 4.287054
```

![predicted_actual](images\predicted-vs-actual.png)

### Predicted VS Actual Fed Funds Rate using RandomForest

```text
Predicted Vs Actual Fed Funds Rate

     Actual  Predicted
5      1.28    0.93660
7      1.29    1.35300
10     1.43    1.48950
23     2.71    2.73705
29     2.94    2.90800
..      ...        ...
835    5.33    5.33000
838    5.33    5.33000
847    4.33    4.36855
848    4.33    4.35925
850    4.33    4.34725

[171 rows x 2 columns]
    Actual  Predicted
5     1.28    0.93660
7     1.29    1.35300
10    1.43    1.48950
23    2.71    2.73705
29    2.94    2.90800
30    2.84    2.94405
31    3.00    2.96380
33    3.00    2.99120
39    3.50    3.38165
49    1.53    1.49535
54    2.48    2.44325
63    3.98    3.77050
65    3.99    3.88970
66    3.99    3.87935
67    3.97    3.86185
72    3.23    3.26270
76    2.44    2.52375
77    1.98    2.44045
78    1.45    2.35585
81    1.49    2.03595
     Actual  Predicted
744    0.39    0.38405
750    0.65    0.60150
751    0.66    0.80325
765    1.69    1.62735
767    1.82    1.85045
773    2.27    2.24560
786    1.55    1.49260
792    0.09    0.08960
800    0.07    0.07730
804    0.10    0.08040
806    0.08    0.08220
815    1.21    0.83275
816    1.68    1.82865
818    2.56    2.96185
828    5.12    5.13185
835    5.33    5.33000
838    5.33    5.33000
847    4.33    4.36855
848    4.33    4.35925
850    4.33    4.34725
```



