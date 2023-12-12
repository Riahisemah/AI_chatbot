import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

def load_json_data(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

def load_txt_data(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    conversations = []
    for i in range(0, len(lines), 2):
        user_input = lines[i].strip()
        bot_response = lines[i + 1].strip()
        conversations.append({'user_input': user_input, 'bot_response': bot_response})

    return {'conversations': conversations}


# Choose the data source (JSON or TXT)
file_type = "json"  # Change to "txt" if using a text file

if file_type == "json":
    intents = load_json_data('intents.json')
elif file_type == "txt":
    intents = load_txt_data('conversations.txt')  # Replace with your TXT file name
else:
    print("Invalid file type specified.")
    exit()

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents'] if file_type == "json" else intents['conversations']:
    for pattern in intent['patterns'] if file_type == "json" else intent['user_input']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag'] if file_type == "json" else intent['bot_response']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag =[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('Done')
