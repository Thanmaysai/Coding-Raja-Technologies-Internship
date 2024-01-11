import json
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources if needed
# nltk.download('punkt')

# Load intents from a file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Extract data from intents
patterns = []
responses = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        responses.append(intent['tag'])

# Tokenization and preprocessing using NLTK
word_tokenizer = nltk.tokenize.WordPunctTokenizer()

tokenized_patterns = [word_tokenizer.tokenize(pattern.lower()) for pattern in patterns]

# Create vocabulary
input_vocab = sorted(set(word for pattern in tokenized_patterns for word in pattern))
target_vocab = sorted(set(response for response in responses))

input_word2idx = {word: idx for idx, word in enumerate(input_vocab)}
target_word2idx = {word: idx for idx, word in enumerate(target_vocab)}
idx2target_word = {idx: word for word, idx in enumerate(target_word2idx)}

# Convert texts to sequences of indices
input_sequences = [
    [input_word2idx[word] for word in pattern]
    for pattern in tokenized_patterns
]
target_sequences = [
    target_word2idx[response]
    for response in responses
]

max_input_length = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences, maxlen=max_input_length, padding='post'
)

# Define a simple sequence-to-sequence model
embedding_dim = 128
units = 256

encoder_inputs = tf.keras.layers.Input(shape=(max_input_length,))
encoder_embedding = tf.keras.layers.Embedding(len(input_vocab), embedding_dim)(encoder_inputs)
encoder = tf.keras.layers.LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(len(target_vocab), embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(len(target_vocab), activation='softmax')
output = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model (provide input and target sequences)
model.fit(
    [input_sequences, input_sequences],  # Example, providing same input and target
    np.array(target_sequences),
    batch_size=64,
    epochs=10,
    validation_split=0.2
)


# Function to generate response based on user input
def get_response(user_input):
    tokenized_input = word_tokenizer.tokenize(user_input.lower())
    input_seq = [input_word2idx.get(word, 0) for word in tokenized_input]
    input_seq = pad_sequences([input_seq], maxlen=max_input_length, padding='post')

    prediction = model.predict([input_seq, input_seq])
    predicted_idx = np.argmax(prediction)
    generated_response = idx2target_word.get(predicted_idx, 'Unknown')

    return generated_response


# Example usage:
user_input = "Hi"
response = get_response(user_input)
print(response)

# Integration of user interaction loop
print("Bot: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)
    print(f"Bot: {response}")
