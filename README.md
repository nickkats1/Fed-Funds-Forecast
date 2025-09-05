### Forecasting Fed Funds Rate Through Fred

### Requirements
```bash
pip install fredapi torch torchvision torchaudio numpy pandas sciki-learn matplotlib seaborn catboost xgboot-cpu mlflow
```
Use Fred api key

```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

"""Interest Rates(Fed Funds Rate) """

ffr = fred.get_series("FEDFUNDS")
ffr.name = "Fed Funds Rate"

data = pd.DataFrame(ffr).dropna()
```



![federal_funds_rate_plot](images/fed-funds-rate.png)



### Structure
This will be setup pretty much the same as most timeseries LSTM's via torch.
```python

data.dropna(inplace=True)
training = data.iloc[:,0:1].values



train_split = int(len(training) * .86)
train_data = training[:train_split]
test_data = training[train_split:]
print(f' Shape of training data: {train_data.shape}')
print(f' Shape of testing data: {test_data.shape}')

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def slider(dataframe, seq_length):
    X, y = [], []
    for i in range(len(dataframe) - seq_length):
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
        out, _ = self.lstm(X,(h0, c0))
        out = self.fc(out[:,-1,:])
        return out



bidirectional_lstm = BiLSTM(input_size=1,hidden_size=256,num_layers=2,output_size=1)
epochs = 500
learning_rate = 0.001
bilistm_optimizer = torch.optim.Adam(params=bidirectional_lstm.parameters(),lr=learning_rate)
loss_fn = nn.MSELoss()
```




### Predicted Vs Actual Fed Funds Rate (BiLSTM)

```text
Date  Actual Fed Funds Rate  Predicted Fed Funds Rate
735 2015-10-01                   0.12                  0.101384
736 2015-11-01                   0.12                  0.080849
737 2015-12-01                   0.24                  0.080849
738 2016-01-01                   0.34                  0.204033
739 2016-02-01                   0.38                  0.306643
740 2016-03-01                   0.36                  0.347676
741 2016-04-01                   0.37                  0.327161
742 2016-05-01                   0.37                  0.337419
743 2016-06-01                   0.38                  0.337419
744 2016-07-01                   0.39                  0.347676
745 2016-08-01                   0.40                  0.357934
746 2016-09-01                   0.40                  0.368190
747 2016-10-01                   0.40                  0.368190
748 2016-11-01                   0.41                  0.368190
749 2016-12-01                   0.54                  0.378447
750 2017-01-01                   0.65                  0.511744
751 2017-02-01                   0.66                  0.624480
752 2017-03-01                   0.79                  0.634727
753 2017-04-01                   0.90                  0.767890
754 2017-05-01                   0.91                  0.880513
          Date  Actual Fed Funds Rate  Predicted Fed Funds Rate
834 2024-01-01                   5.33                  5.368036
835 2024-02-01                   5.33                  5.368036
836 2024-03-01                   5.33                  5.368036
837 2024-04-01                   5.33                  5.368036
838 2024-05-01                   5.33                  5.368036
839 2024-06-01                   5.33                  5.368036
840 2024-07-01                   5.33                  5.368036
841 2024-08-01                   5.33                  5.368036
842 2024-09-01                   5.13                  5.368036
843 2024-10-01                   4.83                  5.167683
844 2024-11-01                   4.64                  4.866721
845 2024-12-01                   4.48                  4.675848
846 2025-01-01                   4.33                  4.514956
847 2025-02-01                   4.33                  4.363991
848 2025-03-01                   4.33                  4.363991
849 2025-04-01                   4.33                  4.363991
850 2025-05-01                   4.33                  4.363991
851 2025-06-01                   4.33                  4.363991
852 2025-07-01                   4.33                  4.363991
853 2025-08-01                   4.33                  4.363991
```



### Predicted VS Actual Fed Funds Rate using RandomForest

```text
Predicted Vs Actual Prices

     Actual  Predicted
23     2.71     2.7406
30     2.84     2.9412
31     3.00     2.9568
33     3.00     2.9914
39     3.50     3.3698
..      ...        ...
833    5.33     5.3300
835    5.33     5.3300
840    5.33     5.3080
849    4.33     4.3300
850    4.33     4.3300

[112 rows x 2 columns]
     Actual  Predicted
23     2.71     2.7406
30     2.84     2.9412
31     3.00     2.9568
33     3.00     2.9914
39     3.50     3.3698
49     1.53     1.3322
63     3.98     3.7896
65     3.99     3.8948
66     3.99     3.8596
67     3.97     3.8460
72     3.23     3.3086
76     2.44     2.5128
77     1.98     2.4636
78     1.45     2.3514
86     1.88     2.1540
96     2.71     2.7352
109    3.49     3.0720
110    3.48     3.4388
120    3.42     3.4950
136    4.10     4.1010
     Actual  Predicted
706    0.11     0.1322
709    0.08     0.0874
713    0.09     0.0772
733    0.14     0.1364
740    0.36     0.3734
746    0.40     0.4034
753    0.90     0.8332
767    1.82     1.8212
778    2.39     2.4098
788    0.65     1.2410
792    0.09     0.0796
802    0.06     0.0732
808    0.08     0.0800
819    3.08     2.8986
830    5.33     5.3216
833    5.33     5.3300
835    5.33     5.3300
840    5.33     5.3080
849    4.33     4.3300
850    4.33     4.3300
```



