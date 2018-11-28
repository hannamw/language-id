import argparse
import pathlib
import pickle
import time
from pathlib  import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(description="specify the number of languages and which languages you want to differentiate between")
parser.add_argument("-p", action = "store", nargs = '?', default = ".", dest = "path", help = "Path to WiLI data")
parser.add_argument("-n", action = "store", nargs = '?', default = "2,2", dest = "ngramrange", help = "input x,y, with no spaces: n-grams for n in range x,y will be used")
parser.add_argument("-d", action = "store", nargs = '?', default = 10, dest = "min_freq", help = "minimum document frequency to include a word in the corpus")
parser.add_argument("-s", action = "store", nargs = '?', default = None, dest = "save", help = "save a copy of the model at the specified path")
parser.add_argument("-l", dest = "languages", nargs = "+", default = ["spa", "arg"])
args = parser.parse_args()

langs = args.languages
nlangs = len(langs)
save = args.save
path = args.path
langMap = {langs[i]:i for i in range(len(langs))}
nmin, nmax = args.ngramrange.split(",")
nmin = int(nmin)
nmax = int(nmax)
minimum_frequency = args.min_freq

x_train = []
y_train = []

x_test = []
y_test = []

print(langs)

def parse_input(affix, x_vals, y_vals):
    train_filepath = Path(path, "x_"+ affix)
    test_filepath = Path(path, "y_"+affix)
    with train_filepath.open(encoding="utf8") as x_values:
        with test_filepath.open(encoding="utf8") as y_values:
            for line, language in zip(x_values, y_values):
                line = line.strip()
                language = language.strip()
                if language in langs:
                    #print("FOUND ONE %s" %language)
                    x_vals.append(line)
                    y_vals.append(langMap[language])

parse_input("train.txt", x_train, y_train)
parse_input("test.txt", x_test, y_test)

cv = CountVectorizer(analyzer = "char_wb", ngram_range = (nmin,nmax), min_df =minimum_frequency)
x_train_vectorized = cv.fit_transform(x_train).toarray()
x_test_vectorized = cv.transform(x_test).toarray()

log_res = LogisticRegression()

start = time.time()
log_res.fit(x_train_vectorized, y_train)
end = time.time()

print("Finished training in %f seconds" %(end - start))

test_score = log_res.score(x_test_vectorized, y_test)

print("The test score is %f"%(test_score))

if(save != None):
    with Path(save).open("wb") as handle:
        pickle.dump(log_res, handle, protocol=pickle.HIGHEST_PROTOCOL)