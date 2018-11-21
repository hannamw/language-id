#extracts from each sentence 10 digraphs, and puts with them their associated language
#the languages are numbered in the order they are input

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

def generate_x_vecs(trainpath, testpath):
    x_trainfile = open(trainpath, "r", encoding = "utf-8")
    x_testfile = open(testpath, "r", encoding = "utf-8")
    cv = CountVectorizer(analyzer = "char_wb", ngram_range = (2,2), min_df =10)
    
    training_vec_array = cv.fit_transform(x_trainfile).toarray()
    
    test_vec_array = cv.transform(x_testfile).toarray()
    
    x_trainfile.close()
    x_testfile.close()
    return [training_vec_array, test_vec_array]
    
    
def main():    
    lang = []

    lang.append(input("Enter the 3-digit language code of the first language: ").strip())
    lang.append(input("Enter the 3-digit language code of the second language: ").strip())
    lang.append(input("Enter the 3-digit language code of the third language: ").strip())

    combined_lang_string = ""

    for language in lang:
        combined_lang_string += language + "_"
    
    clsd = combined_lang_string + "/" + combined_lang_string
    
    
    x_vecs = generate_x_vecs(clsd + "train.txt", clsd + "test.txt")
    x_train = x_vecs[0]
    train_target = open(clsd + "train_target.txt", "r", encoding = "utf-8")
    tr_array = train_target.readlines()
    y_train = [int(line[0]) for line in tr_array]
    
    x_test = x_vecs[1]
    test_target = open(clsd + "test_target.txt", "r", encoding = "utf-8")
    te_array = test_target.readlines()
    y_test = [int(line[0]) for line in te_array]
    
    train_target.close()
    test_target.close()
    
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    print("The random forest classifier achieved a score of " + str(rfc.score(x_test,y_test)))
    
    svc = SVC()
    svc.fit(x_train, y_train)
    print("The SVC achieved a score of " + str(svc.score(x_test, y_test)))
    
main()