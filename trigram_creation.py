training_set = []
with open('imdb_train.csv') as file:
    training_set = file.readlines()

# Using CountVectorizer
Cvectorizer = CountVectorizer(ngram_range=(3, 3))
count_vec_model = Cvectorizer.fit_transform(training_set)

f = open("Cvector_Score.txt", "a")
f.write(str(count_vec_model.toarray()))
f.close()

# Applying TFIDF
Tvectorizer = TfidfVectorizer(ngram_range=(3, 3))
TFIDF_vec_model = Tvectorizer.fit_transform(training_set)
TFIDF_scores = (TFIDF_vec_model.toarray())

f = open("TFIDF_scores.txt", "a")
f.write(str(TFIDF_scores))
f.close()
