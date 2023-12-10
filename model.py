import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation

sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 30

FILE_PATH = "scraper/sevdalinka.csv"
BASENAME = os.path.basename(FILE_PATH)

text = open(FILE_PATH, encoding="utf-8").read()
text = text.lower()

text = text.translate(str.maketrans("", "", punctuation))

n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

# make results folder if does not exist yet
if not os.path.isdir("results"):
    os.mkdir("results")

pickle.dump(char2int, open(f"results/{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"results/{BASENAME}-int2char.pickle", "wb"))

encoded_text = np.array([char2int[c] for c in text])
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
for char in char_dataset.take(8):
    print(char.numpy(), int2char[char.numpy()])

sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)
for sequence in sequences.take(2):
    print(''.join([int2char[i] for i in sequence.numpy()]))

def split_sample(sample):
    # example :
    # sequence_length is 10
    # sample is "python is a great pro" (21 length)
    # ds will equal to ('python is ', 'a') encoded as integers
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample)-1) // 2):
        # first (input_, target) will be ('ython is a', ' ')
        # second (input_, target) will be ('thon is a ', 'g')
        # third (input_, target) will be ('hon is a g', 'r')
        # and so on
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        # extend the dataset with these samples by concatenate() method
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

# prepare inputs and targets
dataset = sequences.flat_map(split_sample)

def one_hot_samples(input_, target):
    # onehot encode the inputs and the targets
    # Example:
    # if character 'd' is encoded as 3 and n_unique_chars = 5
    # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

dataset = dataset.map(one_hot_samples)

for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")

ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

model_weights_path = f"results/{BASENAME}-{sequence_length}.h5"
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
# save the model
model.save(model_weights_path)