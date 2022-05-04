from os import read
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from modelbuilding import logistic, dtree, randomforest, svc, knn


app = Flask(__name__)


# reading the data from csv file

def readdata():
    colnames = ['LIMIT_BAL', 'Gender', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default payment next month']
    df = pd.read_csv(r'CaseStudy_FraudIdentification.csv',header=0)

    return df


# groupby with status col

def getdict(colname):

    df = readdata()
    return dict(df.groupby([df.columns])['default payment next month'].count())

# label encoding


def labelencoding(df):
    encoder = LabelEncoder()
    data = df.copy()
    getmappings = {}
    for col in list(data.columns):
        data[col] = encoder.fit_transform(data[col])

        # get the mappings of the encoded dataframe
        getmappings[col] = dict(
            zip(encoder.classes_, encoder.transform(encoder.classes_)))

    return getmappings, data


@ app.route("/", methods=["GET", "POST"])
def hello_world():

    getmappings, data = labelencoding(readdata())

    if request.method == "POST":
        print("Inside main method")
        mydict = request.form
        LIMIT_BAL= mydict['LIMIT_BAL']
        print(LIMIT_BAL)
        Gender = mydict['Gender']
        print(Gender)
        Education = mydict['Education']
        print(Education)
        Marriage_Status = mydict['marriage']
        print(Marriage_Status)
        age = mydict['age']
        print(age)
        pay_st_jan=mydict['pay_0']
        print(pay_st_jan)
        pay_st_feb=mydict['pay_2']
        print(pay_st_feb)
        pay_st_mar=mydict['pay_3']
        print(pay_st_mar)
        pay_st_april=mydict['pay_4']
        print(pay_st_april)
        pay_st_may=mydict['pay_5']
        print(pay_st_may)
        pay_st_june=mydict['pay_6']
        print(pay_st_june)

        bill_amt_jan=mydict['BILL_AMT1']
        print(bill_amt_jan)
        bill_amt_feb=mydict['BILL_AMT2']
        print(bill_amt_feb)
        bill_amt_mar=mydict['BILL_AMT3']
        print(bill_amt_mar)
        bill_amt_april=mydict['BILL_AMT4']
        print(bill_amt_april)
        bill_amt_may=mydict['BILL_AMT5']
        print(bill_amt_may)
        bill_amt_june=mydict['BILL_AMT6']
        print(bill_amt_june)

        amt_prev_pay_jan=mydict['PAY_AMT1']
        print(amt_prev_pay_jan)
        amt_prev_pay_feb=mydict['PAY_AMT2']
        print(amt_prev_pay_feb)
        amt_prev_pay_mar=mydict['PAY_AMT3']
        print(amt_prev_pay_mar)
        amt_prev_pay_april=mydict['PAY_AMT4']
        print(amt_prev_pay_april)
        amt_prev_pay_may=mydict['PAY_AMT5']
        print(amt_prev_pay_may)
        amt_prev_pay_june=mydict['PAY_AMT6']
        print(amt_prev_pay_june)

        
        algo = mydict['algo']
        print(algo)
        

        # Selection of Algorithm

        algomapper = {'rf': randomforest(
            data), 'dt': dtree(data), 'svc': svc(data)}

        classmapper = {0: 'will default', 1: 'will not default'}
        algorithm = algomapper[algo]
        accuracy, recall, precision, f1score, model = algorithm

        inputparam = [[LIMIT_BAL, Gender, Education, Marriage_Status, age, pay_st_jan,pay_st_feb,
                        pay_st_mar,pay_st_april,pay_st_may, pay_st_june,bill_amt_jan,
                        bill_amt_feb,bill_amt_mar,bill_amt_april,bill_amt_may,bill_amt_june,
                        amt_prev_pay_jan,amt_prev_pay_feb,amt_prev_pay_mar,amt_prev_pay_april,amt_prev_pay_may
                    ,amt_prev_pay_june]]
        
        predict = model.predict(inputparam)
        predictedclass = classmapper[predict[0]]

        return render_template('index.html', predictedclass=predictedclass, display=True, accuracy=round(accuracy*100, 2), precision=precision, showtemplate=True)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
