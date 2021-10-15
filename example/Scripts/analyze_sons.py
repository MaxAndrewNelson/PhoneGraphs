phones = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'Y', 'Z', 'ZH']

sons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0]

phone_to_son = {x:y for x,y in zip(phones, sons)}


class_file = open("HW_type_classes.txt", "r")

for line in class_file:
    print(line)
        