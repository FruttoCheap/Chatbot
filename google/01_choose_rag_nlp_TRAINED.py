import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the data
data = pd.read_csv('training_for_choosing_agent.csv', usecols=['question', 'label'])

# Preprocess the data
tokenizer = Tokenizer(num_words=1000)  # Reduced vocabulary size
tokenizer.fit_on_texts(data['question'])
sequences = tokenizer.texts_to_sequences(data['question'])
padded_sequences = pad_sequences(sequences, maxlen=50)  # Reduced sequence length

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'])

# Split the dataset into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(padded_sequences, labels, test_size=0.3, random_state=42)  # Increased validation split

# Define the model
model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=50),  # Reduced embedding dimensions
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),  # Reduced number of neurons
    Dropout(0.5),  # Added dropout for regularization
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(train_data, train_labels, epochs=50, batch_size=16, validation_data=(val_data, val_labels))  # Reduced epochs and batch size

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(val_data, val_labels)
print(f'Validation accuracy: {accuracy:.2f}')