# information on available features taken from here: https://susanqq.github.io/UTKFace/

race_dict = {
    0 : 'White',
    1 : 'Black',
    2 : 'Asian',
    3 : 'Indian',
    4 : 'Others ', # (like Hispanic, Latino, Middle Eastern)
}

gender_dict = {
    0 : 'Male',
    1 : 'Female',
}


def infer_information_from_filename(filename):
    assert filename.count('_')==3, "File {} broken".format(filename)
    age, gender, race, _ = filename.split("_")
    return int(age), int(gender), int(race)