import pandas as pd
from sklearn.model_selection import train_test_split
import re

def clean(text):
    text = re.sub('\[.*?\]','',text)
    pattern = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    # txt = re.sub(p, '', text)
    # # pattern = re.compile('[^a-z^A-Z^0-9^\u0027]')
    newtext = re.sub(pattern, '', text)
    # pattern2 = re.compile(r'[^\w]')
    # newtext = re.split(pattern2, text)
    return newtext

if __name__ == '__main__':

    data_train = pd.read_csv('./data/train.csv')
    X_train = data_train['ori_sentence'].values.tolist()
    y_train= data_train['bias_labels'].values.tolist()

    data_val = pd.read_csv('./data/valid.csv')
    X_valid = data_val['ori_sentence'].values.tolist()
    y_valid = data_val['bias_labels'].values.tolist()

    X_train, X_, y_train, y_ = train_test_split(X_train,
                                                y_train,
                                                test_size=0.1,
                                                shuffle=True,
                                                stratify=y_train,
                                                random_state=42)

    X_valid, X_,y_valid, y_ = train_test_split(X_valid,
                                                y_valid,
                                                test_size=0.1,
                                                shuffle=True,
                                                stratify=y_valid,
                                                random_state=42)
    traindir = "./data/train.txt"
    validdir = "./data/dev.txt"
    testdir = "./data/test.txt"
    with open(traindir, 'a+', encoding='utf-8-sig') as f:
        for i, j in zip(X_train, y_train):
            i = clean(i)
            f.write(str(i) + '\t' + str(j) + '\n')

    with open(validdir, 'a+', encoding='utf-8-sig') as f:
        for i, j in zip(X_valid, y_valid):
            i = clean(i)
            f.write(str(i) + '\t' + str(j) + '\n')

    with open(testdir, 'a+', encoding='utf-8-sig') as f:
        for i, j in zip(X_valid, y_valid):
            # i = clean(i)
            f.write(str(i) + '\t' + str(j) + '\n')
    f.close()