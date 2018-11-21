#extracts from each sentence 10 digraphs, and puts with them their associated language
#french is 0, english is 1

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

def generate_x_vecs(trainpath, testpath):
    x_trainfile = open(trainpath, "r", encoding = "utf-8")
    x_testfile = open(testpath, "r", encoding = "utf-8")
    cv = CountVectorizer(analyzer = "char_wb", ngram_range = (2,2), min_df =10)
    #cv.fit(x_file)
    #print(cv.vocabulary_)
    training_vec_array = cv.fit_transform(x_trainfile).toarray()
    
    #cvt = CountVectorizer(vocabulary = cv.vocabulary_, analyzer = "char_wb", ngram_range = (2,2), min_df =10) 
    test_vec_array = cv.transform(x_testfile).toarray()
    #print(vec_array)
    #print(cv.vocabulary_)
    x_trainfile.close()
    x_testfile.close()
    return [training_vec_array, test_vec_array]
    
    
def main():    
    lang0 = input("Enter the 3-digit language code of the first language: ").strip()
    lang1 = input("Enter the 3-digit language code of the second language: ").strip()
    combined_lang_string = lang0 + "_" + lang1 + "_"
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