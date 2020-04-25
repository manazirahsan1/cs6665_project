from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
le = preprocessing.LabelEncoder()
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesRegressor as ETRg
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 5, 4
sns.set(style='whitegrid', palette='muted', rc={'figure.figsize': (15, 10)})


# Creating Model
# The Hyper-parameters would be changed as per necessary.
# Mostly, hyper-parameters would be changed based on tuning operations
def define_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):
    # Set seed of randomness for reproduceability
    seed(42)
    tf.random.set_seed(42)
    model = Sequential()  # Model would perform sequentially.
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))  # Setting the hidden layer of the model
    # Setting additional hidden layer if suplied with parameters
    for i in range(1, len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    model.add(Dropout(dr))  # Setting dropout probability supplied as parameters
    # Setting sigmoid activation at output layer
    model.add(Dense(1, activation='sigmoid'))
    # Setting binary cross entropy as loss function of the network
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# Defining function for drawing training/validation error of the trained model
def plot_training_and_validation_accuracy(trained_model):
    # Plot training and validation accuracy against epochs:
    plt.style.use('ggplot')
    plt.plot(trained_model.history['accuracy'], color='green', marker='o', label='Training')
    plt.plot(trained_model.history['val_accuracy'], color='red', marker='d', label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# Defining function for drawing training error of the trained model
def plot_training_accuracy(trained_model):
    # Plot training accuracy against epochs:
    plt.style.use('ggplot')
    plt.plot(trained_model.history['accuracy'], color='green', marker='o', label='Training')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# Getting data: PLEASE PUT THE DATA IN THE ROOT DIRECTORY
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df = pd.concat([train, test], axis=0, sort=True)
# print(len, len(test))

######################################################################################
#                        DATA CLEANSING & FEATURE ENGINEERING                        #
######################################################################################
# Check total and percentage of missing values by column (feature).
total_missing_by_column = df.isnull().sum().sort_values(ascending=False)
percent_missing_by_column = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing_by_column, percent_missing_by_column], axis=1, keys=['Total', 'Percent'])
# missing_data.head(12)

# Creating a TicketId feature, it will tell which person was part of that Family or group
# Creating a new feature TicketId which will select persons in a same family or same group
df_ticket = pd.DataFrame(df.Ticket.value_counts())
df_ticket.rename(columns={'Ticket': 'TicketNum'}, inplace=True)
df_ticket['TicketId'] = pd.Categorical(df_ticket.index).codes
df_ticket.loc[df_ticket.TicketNum < 3, 'TicketId'] = -1
df = pd.merge(left=df, right=df_ticket, left_on='Ticket',
              right_index=True, how='left', sort=False)
df = df.drop(['TicketNum'], axis=1)
# df.head(10)

# Sperating the first name and the last name
df['FamilyName'] = df.Name.apply(lambda x: str.split(x, ',')[0])
# Creating another feature FamilySurv
# If a passenger has no family member accompanied with him then he is assigned to 0.5.
# If a family member is survived or not survived then he is assigned to 1.0 and 0.0 respectively.
df['FamilySurv'] = 0.5
for _, grup in df.groupby(['FamilyName', 'Fare']):
    if len(grup) != 1:
        for index, row in grup.iterrows():
            smax = grup.drop(index).Survived.max()
            smin = grup.drop(index).Survived.min()
            pid = row.PassengerId

            if smax == 1:
                df.loc[df.PassengerId == pid, 'FamilySurv'] = 1.0
            elif smin == 0:
                df.loc[df.PassengerId == pid, 'FamilySurv'] = 0.0
for _, grup in df.groupby(['Ticket']):
    if len(grup) != 1:
        for index, row in grup.iterrows():
            if (row.FamilySurv == 0.0 or row.FamilySurv == 0.5):
                smax = grup.drop(index).Survived.max()
                smin = grup.drop(index).Survived.min()
                pid = row.PassengerId

                if smax == 1:
                    df.loc[df.PassengerId == pid, 'FamilySurv'] = 1.0
                elif smin == 0:
                    df.loc[df.PassengerId == pid, 'FamilySurv'] = 0.0

# Cabin_Number: Finding number of cabins belong to each passenger
df.Cabin = df.Cabin.fillna('0')
regex = re.compile('\s*(\w+)\s*')
df['Cabin_Number'] = df.Cabin.apply(lambda x: len(regex.findall(x)))

df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
mapping = {
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Lady': 'Mrs',
    'Countess': 'Mrs',
    'Dona': 'Mrs',
    'Major': 'Mr',
    'Col': 'Mr',
    'Sir': 'Mr',
    'Don': 'Mr',
    'Jonkheer': 'Mr',
    'Capt': 'Mr',
    'Rev': 'Mr',
    'General': 'Mr'}
df.replace({'Title': mapping}, inplace=True)
# Histogram of suvrivived against title
# train1 = df[0:891].copy()
# sns.set(style="whitegrid")
# plt.figure(figsize=(15,4))
# ax = sns.barplot(x="Title", y="Survived", data=train1, ci=None)

# Let's find simlar data, and fill that for missing fare
# df.loc[(df['Age'] >= 60) & (df['Pclass'] ==3) & (df['Sex'] == 'male') & (df['Embarked'] =='S')]
df.loc[df['Fare'].isnull(), 'Fare'] = 7

# Create a new column fare category (Fare_Category).
df['Fare_Category'] = 0
df.loc[df['Fare'] < 8, 'Fare_Category'] = 0
df.loc[(df['Fare'] >= 8) & (df['Fare'] < 16), 'Fare_Category'] = 1
df.loc[(df['Fare'] >= 16) & (df['Fare'] < 30), 'Fare_Category'] = 2
df.loc[(df['Fare'] >= 30) & (df['Fare'] < 45), 'Fare_Category'] = 3
df.loc[(df['Fare'] >= 45) & (df['Fare'] < 80), 'Fare_Category'] = 4
df.loc[(df['Fare'] >= 80) & (df['Fare'] < 160), 'Fare_Category'] = 5
df.loc[(df['Fare'] >= 160) & (df['Fare'] < 270), 'Fare_Category'] = 6
df.loc[(df['Fare'] >= 270), 'Fare_Category'] = 7
##Histogram of suvrivived against Fare Category
# train_set = df[0:891].copy()
# sns.set(style="whitegrid")
# plt.figure(figsize=(14,3.5))
# ax = sns.barplot(x="Fare_Category", y="Survived",hue='Title', data=train_set, ci=None)

# Creating Family_Size Feature, since big family may survive more likely
df['Family_Size'] = 0
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
# df.Family_Size.value_counts()
# Create a new column fare category (Fare_Category).
df['Alone'] = 0
df.loc[df['Family_Size'] <= 1, 'Alone'] = 1
df.loc[df['Family_Size'] > 1, 'Alone'] = 0
# df.Alone.value_counts()


# Label Encode Totle, Cabin
lsr = {'Title', 'Cabin'}
for i in lsr:
    le.fit(df[i].astype(str))
    df[i] = le.transform(df[i].astype(str))

# Lets predict the age of a person and fill the missing Age
features = ['Pclass', 'SibSp', 'Parch', 'TicketId', 'Fare', 'Cabin_Number', 'Title', 'Alone']
Etr = ETRg(n_estimators=200, random_state=2)
AgeX_Train = df[features][df.Age.notnull()]
AgeY_Train = df['Age'][df.Age.notnull()]
AgeX_Test = df[features][df.Age.isnull()]
# Train and predict the missing ages
Etr.fit(AgeX_Train, np.ravel(AgeY_Train))
AgePred = Etr.predict(AgeX_Test)
df.loc[df.Age.isnull(), 'Age'] = AgePred

##Lets derive AgeGroup feature from age
##New feature Age_Category
df['Age_Category'] = 0
df.loc[(df['Age'] <= 5), 'Age_Category'] = 0
df.loc[(df['Age'] <= 12) & (df['Age'] > 5), 'Age_Category'] = 1
df.loc[(df['Age'] <= 18) & (df['Age'] > 12), 'Age_Category'] = 2
df.loc[(df['Age'] <= 22) & (df['Age'] > 18), 'Age_Category'] = 3
df.loc[(df['Age'] <= 32) & (df['Age'] > 22), 'Age_Category'] = 4
df.loc[(df['Age'] <= 45) & (df['Age'] > 32), 'Age_Category'] = 5
df.loc[(df['Age'] <= 60) & (df['Age'] > 45), 'Age_Category'] = 6
df.loc[(df['Age'] <= 70) & (df['Age'] > 60), 'Age_Category'] = 7
df.loc[(df['Age'] > 70), 'Age_Category'] = 8
# Histogram of Survived against Age_Category
# train_set = df[0:891].copy()
# sns.set(style="whitegrid")
# plt.figure(figsize=(14,3.5))
# ax = sns.barplot(x="Age_Category", y="Survived",data=train_set, ci=None)

# Replace the missing Embarked with value 'C': Most first class passengers came from this port.
df.loc[(df.Embarked.isnull()), 'Embarked'] = 'C'

# Label Encode Embarked, Sex
lst = {'Embarked', 'Sex'}
for i in lst:
    le.fit(df[i].astype(str))
    df[i] = le.transform(df[i].astype(str))

##Scaling up data using Standard scaler
y_train = train['Survived'].values
# select_features = ['Pclass', 'Age', 'Age_Category', 'SibSp', 'Parch', 'Fare',
#                    'Embarked', 'TicketId', 'Cabin_Number', 'Title', 'Cabin',
#                    'Fare_Category', 'Family_Size', 'FamilySurv', 'Sex', 'Alone']
select_features = ['Pclass', 'Age', 'Age_Category', 'SibSp', 'Parch', 'Fare',
                   'Embarked', 'TicketId', 'Cabin_Number', 'Title', 'Cabin',
                   'Fare_Category', 'Family_Size', 'FamilySurv', 'Sex', 'Alone']
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[select_features])
X_train = scaled_df[0:891].copy()
X_test = scaled_df[891:].copy()


####################################################################################
#                           DEEP lEARNING IMPLEMENTATION                           #
####################################################################################
# Cteate model: Use previously defined Function (define_model)
model = define_model()
# Train model:
trained_model = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
val_accuracy = np.mean(trained_model.history['val_accuracy'])
print('Validation accuracy: {}'.format(val_accuracy * 100))
plot_training_accuracy(trained_model)
# plot_training_and_validation_accuracy(trained_model)


# Tuning hyper-parameter:

####################################################################################
#                          Tune: EPOCH & MINI BATCH SIZE                           #
####################################################################################
# Instantiate model
model = KerasClassifier(build_fn=define_model, verbose=0)
# Define the grid search parameters: mini-batch-size and number of epochs
batch_size = [16, 32, 64, 128]
epochs = [20, 50, 80, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
# search the grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
# Predict with tuned parameters:
grid_search_result = grid_search.fit(X_train, y_train)
print('Best accuracy: ', grid_search_result.best_score_)
n_epochs = grid_search_result.best_params_['epochs']
n_batch_size = grid_search_result.best_params_['batch_size']
print('Best number of epochs: {}\nBest mini batch size: {}'.format(n_epochs, n_batch_size))

# Train the model with best params, i.e., mini batch size and number of epochs
# Evaluate its performance
model = define_model()
trained_model = model.fit(X_train, y_train,
                          epochs=n_epochs,
                          batch_size=n_batch_size,
                          validation_split=0.2, verbose=0)
scores = model.evaluate(X_train, y_train)
plot_training_accuracy(trained_model)
# plot_training_and_validation_accuracy(trained_model)



####################################################################################
#                                  Tune: OPTIMIZER                                 #
####################################################################################
# Instantiate model
# Use previously tuned parameters: epochs and mini-batch-size
model = KerasClassifier(build_fn=define_model,
                        epochs=n_epochs,
                        batch_size=n_batch_size,
                        verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
param_grid = dict(opt=optimizer)
# Search the grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2, n_jobs=-1)
# Predict with tuned parameters:
grid_search_result = grid_search.fit(X_train, y_train)
print('Best accuracy: ', grid_search_result.best_score_)
n_optimizer = grid_search_result.best_params_['opt']
print('Best gradient descent optimizer: {}'.format(n_optimizer))

# Train the model with best params, i.e., best optimizer
# Evaluate its performance
model = define_model(opt=n_optimizer)
trained_model = model.fit(X_train, y_train,
                          epochs=n_epochs,
                          batch_size=n_batch_size,
                          validation_split=0.2, verbose=0)
scores = model.evaluate(X_train, y_train)
plot_training_accuracy(trained_model)
# plot_training_and_validation_accuracy(trained_model)


####################################################################################
#                   Tune: ARCHITECTURE (i.e., HIDDEN LAYERS)                       #
####################################################################################
# Instantiate model
# Use previously tuned parameters: epochs, mini-batch-size, optimizer
model = KerasClassifier(build_fn=define_model,
                        epochs=n_epochs,
                        batch_size=n_batch_size,
                        verbose=0)
# define the grid search parameters, i.e., different architecture
layers = [(8), (10), (10, 5), (12, 6), (12, 8, 4)]
param_grid = dict(lyrs=layers)
# search the grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2, n_jobs=-1)
grid_search_result = grid_search.fit(X_train, y_train)
print('Best accuracy: ', grid_search_result.best_score_)
n_hidden_layer = grid_search_result.best_params_['lyrs']
print('Best Hidden Layer: {}'.format(n_hidden_layer))

# Train the model with best params, i.e., best architecture
# Evaluate its performance
model = define_model(opt=n_optimizer, lyrs=n_hidden_layer)
trained_model = model.fit(X_train, y_train,
                          epochs=n_epochs,
                          batch_size=n_batch_size,
                          validation_split=0.2, verbose=0)
scores = model.evaluate(X_train, y_train)
plot_training_accuracy(trained_model)
# plot_training_and_validation_accuracy(trained_model)



####################################################################################
#                          Tune: DROPOUT PROBABILITY                               #
####################################################################################
# Instantiate model
# Use previously tuned parameters: epochs, mini-batch-size, optimizer
model = KerasClassifier(build_fn=define_model,
                        epochs=n_epochs,
                        batch_size=n_batch_size,
                        verbose=0)
# define the grid search parameters
drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
param_grid = dict(dr=drops)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2, n_jobs=-1)
grid_search_result = grid_search.fit(X_train, y_train)
print('Best accuracy: ', grid_search_result.best_score_)
n_dropout_rate = grid_search_result.best_params_['dr']
print('Best dropout rate: {}'.format(n_dropout_rate))

# Train the model with best params, i.e., best dropout probability
# Evaluate its performance
model = define_model(opt=n_optimizer, lyrs=n_hidden_layer, dr=n_dropout_rate)
trained_model = model.fit(X_train, y_train,
                          epochs=n_epochs,
                          batch_size=n_batch_size,
                          validation_split=0.0, verbose=0)
scores = model.evaluate(X_train, y_train)
plot_training_accuracy(trained_model)
# plot_training_and_validation_accuracy(trained_model)
print('Training accuracy: ', trained_model.history['accuracy'][-1])


####################################################################################
#                         FINAL: EXTRACTING PREDICTION                             #
####################################################################################
# Print the best parameters:
print('Best number of epochs: ', n_epochs)
print('Best mini batch size: ', n_batch_size)
print('Best optimizer: ', n_optimizer)
print('Best architecture (hidden layer(s)): ', n_hidden_layer)
print('Best dropout probability: ', n_dropout_rate)

# Predict the performance on test data:
test['Survived'] = model.predict(X_test)
test['Survived'] = test['Survived'].apply(lambda x: round(x, 0)).astype('int')
solution = test[['PassengerId', 'Survived']]
# Extract the prediction
solution.to_csv("Deep_Learning_Solution.csv", index=False)
