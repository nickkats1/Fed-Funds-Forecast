### Forecasting Fed Funds Rate Through Fred

### Requirements
```bash
pip install fredapi torch torchvision torchaudio numpy pandas sciki-learn matplotlib seaborn catboost xgboot-cpu mlflow
```
#### Use Fred api key

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
Root Mean-Squared Error: 0.01003
R2 Score: 99.00%
Mean Absolute Percentage Error: 154102251520.0000
          Date  Actual Fed Funds Rate  Predicted Fed Funds Rate
735 2015-10-01                   0.12                  0.098126
736 2015-11-01                   0.12                  0.077553
737 2015-12-01                   0.24                  0.077553
738 2016-01-01                   0.34                  0.200967
739 2016-02-01                   0.38                  0.303763
740 2016-03-01                   0.36                  0.344869
741 2016-04-01                   0.37                  0.324317
742 2016-05-01                   0.37                  0.334593
743 2016-06-01                   0.38                  0.334593
744 2016-07-01                   0.39                  0.344869
745 2016-08-01                   0.40                  0.355144
746 2016-09-01                   0.40                  0.365419
747 2016-10-01                   0.40                  0.365419
748 2016-11-01                   0.41                  0.365419
749 2016-12-01                   0.54                  0.375694
750 2017-01-01                   0.65                  0.509220
751 2017-02-01                   0.66                  0.622143
752 2017-03-01                   0.79                  0.632406
753 2017-04-01                   0.90                  0.765783
754 2017-05-01                   0.91                  0.878578
          Date  Actual Fed Funds Rate  Predicted Fed Funds Rate
834 2024-01-01                   5.33                  5.369522
835 2024-02-01                   5.33                  5.369522
836 2024-03-01                   5.33                  5.369522
837 2024-04-01                   5.33                  5.369522
838 2024-05-01                   5.33                  5.369522
839 2024-06-01                   5.33                  5.369522
840 2024-07-01                   5.33                  5.369522
841 2024-08-01                   5.33                  5.369522
842 2024-09-01                   5.13                  5.369522
843 2024-10-01                   4.83                  5.169126
844 2024-11-01                   4.64                  4.868084
845 2024-12-01                   4.48                  4.677151
846 2025-01-01                   4.33                  4.516203
847 2025-02-01                   4.33                  4.365180
848 2025-03-01                   4.33                  4.365180
849 2025-04-01                   4.33                  4.365180
850 2025-05-01                   4.33                  4.365180
851 2025-06-01                   4.33                  4.365180
852 2025-07-01                   4.33                  4.365180
853 2025-08-01                   4.33                  4.365180
```



### Predicted VS Actual Fed Funds Rate using RandomForest

```text
Root Mean-Squared Error: 0.3096
R2 Score: 99.01%
Mean Absolute Percentage Error: 0.0570
Predicted Vs Actual Prices

     Actual  Predicted
23     2.71     2.7368
30     2.84     2.9388
31     3.00     2.9564
33     3.00     2.9886
39     3.50     3.4022
49     1.53     1.4958
63     3.98     3.8158
65     3.99     3.9340
66     3.99     3.8898
67     3.97     3.8574
72     3.23     3.3242
76     2.44     2.5294
77     1.98     2.4706
78     1.45     2.3714
86     1.88     2.1608
96     2.71     2.7132
109    3.49     3.0310
110    3.48     3.4350
120    3.42     3.4970
136    4.10     4.0756
     Actual  Predicted
706    0.11     0.1290
709    0.08     0.0890
713    0.09     0.0786
733    0.14     0.1354
740    0.36     0.3744
746    0.40     0.4050
753    0.90     0.8912
767    1.82     1.8088
778    2.39     2.4078
788    0.65     1.2410
792    0.09     0.0810
802    0.06     0.0742
808    0.08     0.0800
819    3.08     2.9784
830    5.33     5.3216
833    5.33     5.3300
835    5.33     5.3300
840    5.33     5.3200
849    4.33     4.3360
850    4.33     4.3300
```



