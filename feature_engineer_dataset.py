import pandas as pd
import utils.data as utild
import sys

"""
Feature engineering Titanic dataset using pandas framework.

Inspired by the great work of Trevor Stephens:
http://trevorstephens.com/kaggle-titanic-tutorial/getting-started-with-r/
"""

def halt():
    sys.exit(0)

# Read the kaggle titanic train and test datasets.
train = pd.read_csv('data/train_kaggle.csv')
test = pd.read_csv('data/test_kaggle.csv')


# Add a Survived column with zeros to test data frame.
df = pd.DataFrame({'Survived': [0] * test.PassengerId.size})
test = pd.concat([test, df], axis=1)


# Concatenate test and train data frames.
bind = pd.concat([train, test], ignore_index=True)


# There are missing values on Age and Fare columns, remove nans and fill with the mean values.
ageDF = utild.nan2mean(bind.Age)
fareDF = utild.nan2mean(bind.Fare)


# Extract last names from Name column.
lastnameL = []
ls = bind.Name.str.split(',', 1).tolist()
for x in xrange(0, len(ls)):
    lastnameL.append(ls[x][0])
lastnameDF = pd.DataFrame(lastnameL)[0]


# Extract titles from Name as Master, Mr, Mrs, etc.
titlenameL = []
ls = bind.Name.str.split('\,(.*?)\.').tolist()
for x in xrange(0, len(ls)):
    titlenameL.append(ls[x][1].strip())
titlenameDF = pd.DataFrame(titlenameL)[0]


# Split ticket codes and ticket numbers.
ticketL = []
ticketnumberL = []
ticketcodeL = []
ls = bind.Ticket.str.split(' ').tolist()
df = pd.DataFrame(ls)
for x in xrange(len(df[0])):
    if df[2][x] != None:
        ticketnumberL.append(df[2][x])
    elif df[1][x] != None:
        ticketnumberL.append(df[1][x])
    elif df[0][x] != None and df[0][x].isdigit():
        ticketnumberL.append(df[0][x])
    else:
        ticketnumberL.append(0)
    if not df[0][x].isdigit():
        ticketcodeL.append(df[0][x])
    else:
        ticketcodeL.append(0)


ticketnumberDF = pd.DataFrame(ticketnumberL)[0]
ticketcodeDF = pd.DataFrame(ticketcodeL)[0]



# Create vector representations for the columns with string values.
# Then take each column and assign a vector for each unique string.
# Then create lists with vector representations for those columns with string values.
# Finally Separate elements of each vector to multiple columns.
cabinDF = utild.uniq2multi(bind.Cabin)
lastnameDF = utild.uniq2multi(lastnameDF, col_name='Lastname')
sexDF = utild.uniq2multi(bind.Sex)
ticketcodeDF = utild.uniq2multi(ticketcodeDF, col_name='Ticketcode')
embarkedDF = utild.uniq2multi(bind.Embarked)
titlenameDF = utild.uniq2multi(titlenameDF, col_name='Nametitle')


# Concat final columns as bind2.
bind2 = pd.concat([
                    bind.PassengerId,
                    bind.Survived,
                    bind.Pclass,
                    titlenameDF,
                    lastnameDF,
                    sexDF,
                    ageDF, 
                    bind.SibSp,
                    bind.Parch,
                    ticketcodeDF, 
                    fareDF,
                    cabinDF,
                    embarkedDF
                    ],axis=1)


# Split final work as test and train again.
train_f, test_f = bind2[:891], bind2[891:]

# Save.
train_f.to_csv('data/train_f_engineered.csv', sep=',', index=False)
test_f.to_csv('data/test_f_engineered.csv', sep=',', index=False)


# Normalize columns
bind2_norm = bind2.copy()
for col in bind2_norm.columns:
    if (col != 'PassengerId' and col != 'Survived'):
        if (col == 'Pclass' or col == 'Age' or col == 'SibSp' 
                            or col == 'Parch' or col == 'Fare'):
            bind2_norm[col] = (
                (bind2_norm[col].astype('float64') 
                    - bind2_norm[col].astype('float64').mean()) 
                / bind2_norm[col].astype('float64').std()
            )

train_f_norm, test_f_norm = bind2_norm[:891], bind2_norm[891:]

train_f_norm.to_csv('data/train_f_engineered_norm.csv', sep=',', index=False)
test_f_norm.to_csv('data/test_f_engineered_norm.csv', sep=',', index=False)