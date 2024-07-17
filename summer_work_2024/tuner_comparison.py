import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch, GridSearch
import time
import csv
import multiprocessing

# # Define the model-building function
# def build_model(hp):
#     model = Sequential()
#     model.add(Flatten(input_shape=(28, 28)))
#     model.add(Dense(units=hp.Choice('units', values=[16, 64, 128]), activation='relu'))
#     model.add(Dense(10, activation='softmax'))
#     model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[0.01, 0.0001])),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

    # Define the model-building function
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load data
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

# Function to run tuner
def run_tuner(distribution_strategy=None,name='my_project'):
    tuner = GridSearch(
        build_model,
        max_trials=6,  # Number of trials to run
        objective='val_accuracy',
        executions_per_trial=1,
        directory='my_dir',
        project_name=name,
        distribution_strategy=distribution_strategy)

    tuner.search_space_summary()
    tuner.search(x=x_train, y=y_train, epochs=5, validation_data=(x_val, y_val))
    return tuner

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()
devices = [f"/cpu:{i}" for i in range(num_cores)]

# Sequential run
start_time = time.time()
sequential_tuner = run_tuner(name='seq')
sequential_duration = time.time() - start_time

# Parallel run with tf.distribute.MirroredStrategy
strategy = tf.distribute.MirroredStrategy(devices=devices)
start_time = time.time()
parallel_tuner = run_tuner(strategy, name='paral')
parallel_duration = time.time() - start_time

# Get the best hyperparameters and validation accuracy for sequential tuner
sequential_best_hp = sequential_tuner.get_best_hyperparameters()[0].values
sequential_best_score = sequential_tuner.oracle.get_best_trials(num_trials=1)[0].score

# Get the best hyperparameters and validation accuracy for parallel tuner
parallel_best_hp = parallel_tuner.get_best_hyperparameters()[0].values
parallel_best_score = parallel_tuner.oracle.get_best_trials(num_trials=1)[0].score

# Write results to CSV in terms of minutes
with open('tuner_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run Type', 'Duration (minutes)','Duration (seconds)', 'Best Hyperparameters', 'Validation Accuracy'])
    writer.writerow(['Sequential', sequential_duration / 60, sequential_duration, sequential_best_hp, sequential_best_score])
    writer.writerow(['Parallel', parallel_duration / 60, parallel_duration, parallel_best_hp, parallel_best_score])

print("Results written to tuner_results.csv")