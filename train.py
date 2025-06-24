import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Custom_Modules.modules as md
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#hyperparameters for our model
ALPHA, BETA = 0.1, 1.0  # Hamiltonian, Regularization
EXJ, DMN, HEXTZ, KZ = 1.0, 0.5, 0.0, 0.5
EPOCHS = 2000
BATCH_SIZE = 2

save_dir = f"models/{ALPHA}_{BETA}/{EXJ}_{DMN}_{HEXTZ}_{KZ}/"

x_train = np.load(
    "circle.npy")  # "train.npy", "circles_skynum.npy", "circle.npy", "square.npy", "triangle.npy"
y_train = np.ones((len(x_train), 1), dtype=np.float32)
x_valid = np.concatenate(
    [np.load("{}".format(path)) for path in ["circle.npy", "square.npy", "triangle.npy"]], axis=0)
y_valid = np.ones((len(x_valid), 1), dtype=np.float32)

os.makedirs(save_dir, exist_ok=True)
if os.listdir(save_dir) == []:
    model_dir = save_dir + "0"
else:
    model_dir = save_dir + str(max([int(path) for path in os.listdir(save_dir)]) + 1)

#bring the model structure
model = md.cnn_model(
    alpha=ALPHA,
    beta=BETA,
    exJ=EXJ,
    DMN=DMN,
    Hextz=HEXTZ,
    Kz=KZ,
)

#uploading the filled model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.nn.summary()
# callbacks = md.get_callbacks(os.path.join(model_dir, "ckpt", "Ep{epoch:04d}"))
callbacks = [md.CustomCallback(model_dir, (x_valid, y_valid))]

#train the model
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_valid, y_valid),
    callbacks=callbacks
)

#save losses
df_history_data = pd.DataFrame(history.history)
df_history_data.index += 1
df_history_data.to_csv(os.path.join(model_dir, "history.csv"))
print("Training Complete")
