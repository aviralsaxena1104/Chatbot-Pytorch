import numpy as np
import random
import json
import nltk
#nltk.download('punkt')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')



# import numpy as np
# import random
# import json
# import nltk
# from nltk_utils import bag_of_words, tokenize, stem
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # Download NLTK data
# nltk.download('punkt')

# # Load intents from JSON file
# with open('intents.json', 'r') as f:
#     intents = json.load(f)

# # Process intents data
# all_words = []
# tags = []
# xy = []
# for intent in intents['intents']:
#     tag = intent['tag']
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         w = tokenize(pattern)
#         all_words.extend(w)
#         xy.append((w, tag))

# ignore_words = ['?', '.', '!']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# # Create training data
# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     bag = bag_of_words(pattern_sentence, all_words)
#     # Ensure bag_of_words returns integers instead of floats
#     bag = [str(word) for word in bag]
#     X_train.append(bag)
#     y_train.append(tag)

# # Convert lists to numpy arrays
# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # Initialize CountVectorizer
# vectorizer = CountVectorizer(vocabulary=all_words)

# # Fit the vectorizer and transform X_train
# X_train_transformed = vectorizer.fit_transform([" ".join(words) for words in X_train])

# # Initialize LabelEncoder
# label_encoder = LabelEncoder()

# # Fit LabelEncoder and transform y_train
# y_train_transformed = label_encoder.fit_transform(y_train)

# # Initialize MLPClassifier
# model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=1000)

# # Train the model
# model.fit(X_train_transformed, y_train_transformed)

# # Save the model and related data
# data = {
#     "model": model,
#     "vectorizer": vectorizer,
#     "label_encoder": label_encoder,
#     "all_words": all_words,
#     "tags": tags
# }

# FILE = "data_sklearn.pkl"
# with open(FILE, 'wb') as f:
#     pickle.dump(data, f)

# print(f'Training complete. Model saved to {FILE}')
