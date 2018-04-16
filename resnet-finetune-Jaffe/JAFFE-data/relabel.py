import csv
import os

# 1) Positive: happy
# 2) Negative: Disgust, Sad, Anger
# 3) Neutral: Neutral
# 4) Surprise: Surprise

# 0"Angry", 1"Disgust", 2"Fear", 3"Happy", 4"Sad", 5"Surprise", 6"Neutral"

emotiondict = {
    0 : 1,
    1 : 1,
    2 : 1,
    3 : 0,
    4 : 1,
    5 : 3,
    6 : 2
}

inputfile = open('test_labels.csv','r')
outputfile = open('new_test_labels.csv','w')

with open('test_labels.csv') as inf, open('new_test_labels.csv', 'w', newline='') as outf:
    reader = csv.reader(inf)
    writer = csv.writer(outf)
    for line in reader:
        if line[0] == '0':
            writer.writerow([1])
        elif line[0] == '1':
            writer.writerow([1])
        elif line[0] == '2':
            writer.writerow([1])
        elif line[0] == '3':
            writer.writerow([0])
        elif line[0] == '4':
            writer.writerow([1])
        elif line[0] == '5':
            writer.writerow([3])
        elif line[0] == '6':
            writer.writerow([2])
        else:
            print ('no match')

    writer.writerows(reader)

# os.remove('path/to/filename')
# os.rename('path/to/filename_temp', 'path/to/filename')

# csv_file = 'training_labels.csv'  # file to be updated
# tempfilename = os.path.splitext(csv_file)[0] + '.bak'
# try:
#     os.remove(tempfilename)  # delete any existing temp file
# except OSError:
#     pass
# os.rename(csv_file, tempfilename)

# # create a temporary dictionary from the input file
# with open(tempfilename, mode='r') as infile:
#     reader = csv.reader(infile, skipinitialspace=True)
#     header = next(reader)  # skip and save header
#     temp_dict = {row[0]: row[1] for row in reader}

# # only add items from my_dict that weren't already present
# temp_dict.update({key: value for (key, value) in mydict.items()
#                       if key not in temp_dict})

# # create updated version of file
# with open(csv_file, mode='wb') as outfile:
#     writer = csv.writer(outfile)
#     writer.writerow(header)
#     writer.writerows(temp_dict.items())

# os.remove(tempfilename) 