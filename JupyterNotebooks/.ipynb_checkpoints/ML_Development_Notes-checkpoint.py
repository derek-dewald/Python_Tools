# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error

def BinaryColumnCreator(df,
                        column_name,
                        new_column_name,
                        value,
                        calculation):
  
  if calculation=='>':
    df[new_column_name] = np.where(df[column_name]>value,1,0)
  elif calculation =='=':
    df[new_column_name] = np.where(df[column_name]==value,1,0)
  elif calculation =='<':
    df[new_column_name] = np.where(df[column_name]<value,1,0)
  elif calculation =='isin':
    df[new_column_name] = np.where(df[column_name].isin(value),1,0)
  elif calculation =='contains':
    df[new_column_name] = np.where(df[column_name].str.contains(value),1,0)
  elif calculation == 'dict':
    for key,value in value.items():
        try:
          df[new_column_name] = np.where((df[key]==value)&(df[new_column_name]==1),1,0)
        except:
          df[new_column_name] = np.where(df[key]==value,1,0)


records = []

for member_id in range(0,1000):
  deposit_val = np.random.randint(5000,100000)
  lending_val = np.random.randint(5000,100000)
  tran_val = np.random.randint(1000,10000)
  tran_vol = tran_val / 100

  for month in range(1,13):
    deposit_val = deposit_val * np.random.uniform(.15, .8)
    lending_val = lending_val * np.random.uniform(.15, .8)
    tran_val = round(tran_val * np.random.uniform(.15, .8))
    
    if tran_val>0:
      tran_vol = round(max(1,tran_val/100),0)
    else:
      tran_vol = 0

    records.append({'MEMBERNBR':member_id,
                    'MONTH':month,
                    'DEPOSIT_VAL':deposit_val,
                    'LENDING_VAL':lending_val,
                    'TRAN_VAL':tran_val,
                    'TRAN_VOL':tran_vol})

pd.set_option('display.float_format', '{:.2f}'.format)

df = pd.DataFrame(records)

BinaryColumnCreator(df,'DEPOSIT_VAL','DEPOSIT_FLAG',1000,'>')
BinaryColumnCreator(df,'LENDING_VAL','LENDING_FLAG',1000,'>')
BinaryColumnCreator(df,'TRAN_VAL','TRAN_VAL_FLAG',500,'>')
BinaryColumnCreator(df,'TRAN_VOL','TRAN_VOL_FLAG',2,'>')
BinaryColumnCreator(df,"",'ATTRITION_FLAG',{'DEPOSIT_FLAG':0,'LENDING_FLAG':0,'TRAN_VAL_FLAG':0,'TRAN_VOL_FLAG':0},'dict')


print(df['DEPOSIT_FLAG'].sum())
print(df['LENDING_FLAG'].sum())
print(df['TRAN_VAL_FLAG'].sum())
print(df['TRAN_VOL_FLAG'].sum())
print(df['ATTRITION_FLAG'].sum())

df['ATTRITION_FLAG_LAG'] = df.groupby('MEMBERNBR')['ATTRITION_FLAG'].shift(1).fillna(0)

df

new_df = df[['MEMBERNBR']].drop_duplicates('MEMBERNBR')

for month in df['MONTH'].unique():
  if len(new_df)==0:
    new_df = df[df['MONTH']==month][['MEMBERNBR','DEPOSIT_VAL','LENDING_VAL','TRAN_VAL','TRAN_VOL','ATTRITION_FLAG']].copy()
    new_df = new_df.rename(columns={x:x if x=='MEMBERNBR' else f"{x}_{month}M" for x in temp_df.columns})

  else:
    temp_df = df[df['MONTH']==month][['MEMBERNBR','DEPOSIT_VAL','LENDING_VAL','TRAN_VAL','TRAN_VOL','ATTRITION_FLAG']].copy()
    temp_df = temp_df.rename(columns={x:x if x=='MEMBERNBR' else f"{x}_{month}M" for x in temp_df.columns})
    new_df = new_df.merge(temp_df,on='MEMBERNBR',how='left')

new_df = new_df.merge(df[df['MONTH']==12][['MEMBERNBR','ATTRITION_FLAG']],on='MEMBERNBR',how='left')



df2 = pd.DataFrame()

for month_int in range(1,12):
    d = new_df[['MEMBERNBR',f'DEPOSIT_VAL_{month_int}M',f'LENDING_VAL_{month_int}M',f'TRAN_VAL_{month_int}M',f'TRAN_VOL_{month_int}M',f'ATTRITION_FLAG_{month_int}M',f'DEPOSIT_VAL_{month_int+1}M',f'LENDING_VAL_{month_int+1}M',f'TRAN_VAL_{month_int+1}M',f'TRAN_VOL_{month_int+1}M',f'ATTRITION_FLAG_{month_int+1}M']].rename(columns={f'DEPOSIT_VAL_{month_int}M':'DEPOSIT_VAL_1M',f'LENDING_VAL_{month_int}M':"LENDING_VAL_1M",f'TRAN_VAL_{month_int}M':'TRAN_VAL_1M',f'TRAN_VOL_{month_int}M':'TRAN_VOL_1M',f'ATTRITION_FLAG_{month_int}M':'ATTRITION_FLAG_1M',f'DEPOSIT_VAL_{month_int+1}M':'DEPOSIT_VAL_2M',f'LENDING_VAL_{month_int+1}M':"LENDING_VAL_2M",f'TRAN_VAL_{month_int+1}M':'TRAN_VAL_2M',f'TRAN_VOL_{month_int+1}M':'TRAN_VOL_2M',f'ATTRITION_FLAG_{month_int+1}M':'ATTRITION_FLAG_2M'})
    df2 = pd.concat([df2,d])

df3 = df2[df2['ATTRITION_FLAG_1M']==0]


def MLManualPipeline(df,
                     X_Cols,
                     y_Col,
                     scaler='MinMaxScaler',
                     model_list=['Linear Regression'],
                     test_size=.3,
                     random_state=42):

    if len(X_Cols) == 0:
        X = np.array(df.drop(y_Col,axis=1).copy())
    else:
        X = np.array(df[X_Cols])
    
    y = df[y_Col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scaler =='MinMaxScaler':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    if len(model_list)==0:
        return X_train,X_test,y_train,y_test
    
    else:
        for model in model_list:
            if model == 'Linear Regression':
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred_lr = lr.predict(X_test)
                
            elif model =='Logistic Regression':
                logreg=LogisticRegression()
                logreg.fit(X_train, y_train)
                y_pred = logreg.predict(X_test)
                print(f"Logisitic Regression Model:\n{confusion_matrix(y_test, y_pred)}\n{classification_report(y_test, y_pred)})")

            elif model =='Random Forest':
                
                ############################################ ESTIMATORS
                rf = RandomForestClassifier(random_state=random_state, n_estimators=25)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                print(f"Random Forest with 25 Nodes?>?>?>:\n{confusion_matrix(y_test, y_pred_rf)}\n{classification_report(y_test, y_pred_rf)})")

MLManualPipeline(df=df3.drop(['MEMBERNBR','ATTRITION_FLAG_1M'],axis=1),
                 X_Cols="",
                 y_Col='ATTRITION_FLAG_2M',
                model_list=['Logistic Regression','Random Forest'])


# Preprocessing: Pivot data into 3D array (members, time steps, features)
features = ['DEPOSIT_VAL', 'LENDING_VAL', 'TRAN_VAL', 'TRAN_VOL']
target = 'ATTRITION_FLAG'

# Group by Member_ID and pivot into 3D array
grouped = df.groupby('MEMBERNBR')
X = np.array([grouped[features].get_group(i).values for i in df['MEMBERNBR'].unique()])
y = np.array([grouped[target].get_group(i).values[-1] for i in df['MEMBERNBR'].unique()])

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predict and display results
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
print("Predictions:", y_pred_classes.flatten())
print("Actual:", y_test)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
