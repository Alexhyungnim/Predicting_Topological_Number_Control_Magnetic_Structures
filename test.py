import tensorflow as tf
import numpy as np
from Custom_Modules.modules import compute_skyrmion_number, spin_to_rgb
import Custom_Modules.modules as md
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#hyperparameters
ALPHA, BETA = 0.1, 1.0  # Hamiltonian, Regularization
EXJ, DMN, HEXTZ, KZ = 1.0, 0.5, 0.0, 0.05
EPOCH = 1999
BATCH_SIZE = 1
MODNUM = 1

# data loading
x_test = np.load("Data/ordinary_train/circle.npy").astype(np.float32)[0:1]  # "circle.npy", "triangle.npy", "beehive.npy", "maze.npy", "heart.npy"
plt.imshow(x_test[0], cmap = 'gray', origin = 'lower')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("is")
# x_test += tf.random.normal(tf.shape(x_test), stddev=0.1)
y_test = 1

save_dir = f"models/{ALPHA}_{BETA}/{EXJ}_{DMN}_{HEXTZ}_{KZ}/"
model_dir = save_dir + str(MODNUM)
model = tf.keras.models.load_model(model_dir + "/model")
model.load_weights(model_dir + "/ckpt/Ep{:07d}".format(EPOCH))
vfield = model(x_test[0:1])
print(np.shape(vfield))
pred = compute_skyrmion_number(tf.math.l2_normalize(vfield, axis=-1), padding='periodic')

norm_output = tf.norm(vfield, axis=-1)
reg_loss = tf.reduce_mean(tf.square((tf.ones_like(norm_output) - norm_output)))
energy_loss = tf.reduce_mean(md.get_energy(tf.math.l2_normalize(vfield, axis=-1), exJ=EXJ, DMN= DMN, Kz=KZ, padding='periodic'))
MSE = tf.square(float(y_test) - float(pred))
print("model's prediction:", float(pred), "Reg_Loss:", float(reg_loss), "Energy_Loss :", float(energy_loss), "mse:", float(MSE))

#save directories
output_dir = save_dir + f"{MODNUM}/" + "test_output/"
os.makedirs(output_dir, exist_ok=True)
density_dir = save_dir + f"{MODNUM}/" + "density/"
os.makedirs(density_dir, exist_ok=True)

vfield = model(x_test[0:1])
print(np.shape(vfield))
pred = compute_skyrmion_number(tf.math.l2_normalize(vfield, axis=-1), padding='periodic')
norm_output = tf.norm(vfield, axis=-1)
reg_loss = tf.reduce_mean(tf.square((tf.ones_like(norm_output) - norm_output)))
energy_loss = tf.reduce_mean(md.get_energy(tf.math.l2_normalize(vfield, axis=-1), exJ=EXJ, DMN=0.0, Kz=KZ, padding='periodic'))
MSE = tf.square(float(y_test) - float(pred))

print("model's prediction:", float(pred), "Reg_Loss:", float(reg_loss), "Energy_Loss :", float(energy_loss), "mse:", float(MSE))
plt.imsave(output_dir +f"/{EXJ}_{DMN}_{KZ}.png", spin_to_rgb(vfield[0]), origin = 'lower')
density = md.get_solid_angle_density(tf.math.l2_normalize(vfield, axis=-1), padding='periodic')
plt.imshow(np.squeeze(density[0], axis=-1), vmin=-0.1, vmax=0.1, cmap='gray')
# cbar = plt.colorbar()  # Adds the colorbar
# cbar.set_ticks([-0.1, 0.1])
# cbar.ax.tick_params(labelsize=35)
plt.savefig(density_dir + f"/{EXJ}_{DMN}_{KZ}.png")
plt.close()  # Close the figure to free memory

