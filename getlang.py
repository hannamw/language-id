#extract only certain languages from the "wili data files
#create new .txt files containing those languages only
#for the purpose of 1v1 linguistic comparison
#

import os

lang0 = input("Enter the 3-digit language code of the first language: ").strip()

lang1 = input("Enter the 3-digit language code of the second language: ").strip()

combined_lang_string = lang0 + "_" + lang1 + "_"

os.makedirs(combined_lang_string, exist_ok=True)

clsd = combined_lang_string + "/" + combined_lang_string

x_train = open("wili-2018/x_train.txt", "r",encoding = "utf-8")
x_test = open("wili-2018/x_test.txt", "r",encoding = "utf-8")
y_train = open("wili-2018/y_train.txt", "r",encoding = "utf-8")
y_test = open("wili-2018/y_test.txt", "r",encoding = "utf-8")

filtered_train = open(clsd + "train.txt", "w",encoding = "utf-8")
filtered_test = open(clsd + "test.txt", "w",encoding = "utf-8")
filtered_train_target = open(clsd + "train_target.txt", "w",encoding = "utf-8")
filtered_test_target = open(clsd + "test_target.txt", "w",encoding = "utf-8")


for line1 in x_train:
    this_line = y_train.readline().strip()
    if(this_line == lang0 or this_line == lang1):
        filtered_train.write(line1)
        filtered_train_target.write(str(1 if this_line == lang1 else 0)+"\n")

for line1 in x_test:
    this_line = y_test.readline().strip()
    if(this_line == lang0 or this_line == lang1):
        filtered_test.write(line1)
        filtered_test_target.write(str(1 if this_line == lang1 else 0)+"\n")
        
x_test.close()
x_train.close()
y_test.close()
y_train.close()
filtered_test.close()
filtered_train.close()