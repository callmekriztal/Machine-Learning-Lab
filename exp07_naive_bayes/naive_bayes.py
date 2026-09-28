from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer

news = fetch_20newsgroups(
		subset='all',
		remove=('headers','footers','quotes')
	)

X = news.data
y = news.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = CountVectorizer(stop_words='english',min_df=2)

X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_counts,y_train)

mnb_pred = mnb.predict(X_test_counts)

mnb_acc = accuracy_score(y_test,mnb_pred)
mnb_f1 = f1_score(y_test,mnb_pred,average='macro')

print("Multinomial")
print("Accuracy = ",mnb_acc)
print("F1_Score = ",mnb_f1)

binary_vectorizer = CountVectorizer(stop_words='english',binary=True)

X_train_binary = binary_vectorizer.fit_transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

bnb = BernoulliNB()
bnb.fit(X_train_binary,y_train)

bnb_pred = bnb.predict(X_test_binary)

bnb_acc = accuracy_score(y_test,bnb_pred)
bnb_f1 = f1_score(y_test,bnb_pred,average='macro')

print("\nBernoulli")
print("Accuracy = ",bnb_acc)
print("F1_Score = ",bnb_f1)

print("\nPerformance")

if(mnb_acc > bnb_acc):
	print("Multinomial Model is better")
else:
	print("Bernoulli Model is better")
