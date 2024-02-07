# information on available features taken from here: https://github.com/dchen236/FairFace/blob/master/predict.py

race_dict = {
    0 : 'White',
    1 : 'Black',
    2 : 'Latino_Hispanic',
    3 : 'East Asian',
    4 : 'Southeast Asian',
    5 : 'Indian',
    6 : 'Middle Eastern',
}

gender_dict = {
    0 : 'Male',
    1 : 'Female',
}

age_dict = {
    0 : '0-2',
    1 : '3-9',
    2 : '10-19',
    3 : '20-29',
    4 : '30-39',
    5 : '40-49',
    6 : '50-59',
    7 : '60-69',
    8 : 'more than 70',
}

# and the inverse key-value mapping
race_dict_inv = dict((v,k) for k,v in race_dict.items())
gender_dict_inv = dict((v,k) for k,v in gender_dict.items())
age_dict_inv = dict((v,k) for k,v in age_dict.items())

