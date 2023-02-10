import spacy  # importing spacy
nlp = spacy.load('en_core_web_sm')

# First example
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Second example
tokens = nlp("cat apple monkey banana")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Third example
sentence_to_compare = "Why is my cat on the car"
sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I\'ve lost my car in my car",
    "I\'d like my boat back",
    "I will name my dog Diana"
    ]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Fourth example
tokens = nlp("car bicycle wheelchair plane")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# interesting facts about similarities between cat, monkey, apple and banana
'''
Similarities between cat and monkey (both animals) or apple and banana (both fruit)
are significantly larger than those of an animal (like cat) and a fruit (like apple).
Interestingly there is a larger similarity between monkey and banana (good ol' monkey like banana)
than between cat and banana (cat no like banana)
'''

# interesting facts about similarities between car, bicycle, wheelchair and plane
'''
All of the items have some similarity between them - they are all forms of transport
The most similar item pairs are bicycle/car and bicycle/wheelchair
Bicycle, car and wheelchair are all forms of transport on land
Wheelchair/bicycle are both forms of transport usually by human mechanical input and both have (usually) two wheels
Bicycle/car are more likely to be found on the street, than a wheelchair (more of a pedestrian road transport),
that is why cars are less similar to wheelchairs
Planes fly, rather than roll, so they are less similar to the other three items, but are more similar to cars
than to wheelchairs and bicycles as both use external power to move (combustion engine, electric engine, etc.)
'''

# looking at cat, monkey, apple and banana similarities with a simple language model
'''
Now running a similarity check give very different results.
Word pairs cat/apple and monkey/apple have the highest similarity coefficients (maybe the saying is
apple doesn't fall far from the cat (and/or monkey) ??)
Monkey no longer like banana and cat like banana even less
Cat and monkey are still pretty similar compared to previous results on a more sophisticated language model
Turns out banana and apple are very different ¯\_ (ツ)_/¯
'''