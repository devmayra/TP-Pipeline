'''Código:'''
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def quitarStopwords_eng(texto):
    ingles = stopwords.words("english")
    texto_limpio = [w.lower() for w in texto if w.lower() not in ingles 
                    and w not in string.punctuation 
                    and w not in ["'s", '|', '--', "''", "``"] ]
    return texto_limpio

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lematizar(texto):
    texto_lema = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in texto]
    return texto_lema

lemmatizer = WordNetLemmatizer()

corpus = [
lematizar(quitarStopwords_eng(word_tokenize("Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-"))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-"))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is slower than CPlus and Rust due to its interpreted nature."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-"))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-"))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript have large communities and an extensive number of available libraries."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers.")))
]

corpus_final = []

for oracion in corpus:
    resultado = ' '.join(oracion)
    corpus_final.append(resultado)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_final)
print("Corpus limpio:")
print(corpus_final)
print("\nMatriz TF-IDF:")
print(X.toarray())
print("\nVocabulario:")
print(vectorizer.get_feature_names_out())

'''Análisis:'''
from nltk import FreqDist

todo = []

for oracion in corpus:
    for palabra in oracion:
        todo.append(palabra)

frecuencia= FreqDist(todo)

print("\nLas 6 palabras más utilizada:")
print(frecuencia.most_common(6))

print("\nLa palabra menos utilizada:")
print(frecuencia.most_common()[-1])

print("\nPalabra más repetida por oración:")
for i, oracion in enumerate(corpus):
    freq = FreqDist(oracion)
    palabra, cantidad = freq.most_common(1)[0]
    if cantidad >1:
        print(f"Oración {i+1}: '{palabra}' aparece {cantidad} veces")

'''Gráfico de frecuencias:'''
(frecuencia.plot(20))
plt.show()