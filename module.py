import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Iterable, Union
import tensorflow.keras.backend as K

# to use periodic boundary condition
class PeriodicPadding(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, inputs, *args, **kwargs):
        x = tf.concat([inputs[:, -self.padding:], inputs, inputs[:, :self.padding]], axis=1)
        x = tf.concat([x[:, :, -self.padding:], x, x[:, :, :self.padding]], axis=2)
        return x
class AddNoise(tf.keras.layers.Layer):
    def __init__(self, noise_level=0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level

    def call(self, inputs, *args, **kwargs):
        return inputs * (1. - self.noise_level) + tf.random.normal(tf.shape(inputs), stddev=self.noise_level)

# to make the spin size == 1
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, *args, **kwargs):
        return tf.math.l2_normalize(inputs, axis=-1)
# to map the spin in to color channels(r, g, b)
def spin_to_rgb(X):
  # hsv -> rgb
    def hsv2rgb(hsv):
        hsv = np.asarray(hsv)
        if hsv.shape[-1] != 3: raise ValueError(
            "Last dimension of input array must be 3; " "shape {shp} was found.".format(shp=hsv.shape))
        in_shape = hsv.shape
        hsv = np.array(hsv, copy=False, dtype=np.promote_types(hsv.dtype, np.float32), ndmin=2)

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        r, g, b = np.empty_like(h), np.empty_like(h), np.empty_like(h)

        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        idx = i % 6 == 0
        r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]

        idx = i == 1
        r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]

        idx = i == 2
        r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]

        idx = i == 3
        r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]

        idx = i == 4
        r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]

        idx = i == 5
        r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

        idx = s == 0
        r[idx], g[idx], b[idx] = v[idx], v[idx], v[idx]

        rgb = np.stack([r, g, b], axis=-1)
        return rgb.reshape(in_shape)

    sxmap, symap, szmap = np.split(X, 3, axis=-1)
    szmap =-szmap
    H = np.clip(-np.arctan2(sxmap, -symap) / (2 * np.pi) + 0.5, 0, 1)
    S = np.clip(-szmap + 1., 0., 1.)
    V = np.clip(szmap + 1., 0., 1.)

    img = np.concatenate((H, S, V), axis=-1)
    for i, map in enumerate(img): img[i] = hsv2rgb(map)
    return img


#to acquire the solid angle
def get_solid_angle_density(spin_map, padding='periodic'):
    # spin_map.shape must be (batch, height, width, channels) or (height, width, chennels)
    # return shape of (batch, height, width, 0) or (height, width, 0)
    if padding == 'periodic':
        padded = tf.concat([spin_map, spin_map[..., :1, :, :]], axis=-3)
        padded = tf.concat([padded, padded[..., :1, :]], axis=-2)
    elif padding == 'valid':
        padded = spin_map
    else:
        raise
    spin_map_a = padded[..., :-1, :-1, :]
    spin_map_b = padded[..., :-1, 1:, :]
    spin_map_c = padded[..., 1:, :-1, :]
    spin_map_d = padded[..., 1:, 1:, :]
    # spin_map_a = spin_map
    # spin_map_b = tf.concat([spin_map[..., 0:, :, :], spin_map[..., 0:0, :, :]], axis=-3)
    # spin_map_c = tf.concat([spin_map[..., :, 0:, :], spin_map[..., :, 0:0, :]], axis=-2)
    # spin_map_d = tf.concat([spin_map_b[..., :, 0:, :], spin_map_b[..., :, 0:0, :]], axis=-2)

    absabc = tf.reduce_sum(tf.multiply(spin_map_a, tf.linalg.cross(spin_map_b, spin_map_c)), axis=-1)
    absdbc = tf.reduce_sum(tf.multiply(spin_map_d, tf.linalg.cross(spin_map_b, spin_map_c)), axis=-1)

    omega1 = 2 * tf.atan(absabc / (1. + tf.reduce_sum(tf.multiply(spin_map_a, spin_map_b), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_a, spin_map_c), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_b, spin_map_c), axis=-1)))
    omega2 = 2 * tf.atan(absdbc / (1. + tf.reduce_sum(tf.multiply(spin_map_d, spin_map_b), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_d, spin_map_c), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_b, spin_map_c), axis=-1)))
    return (omega1 - omega2)[..., tf.newaxis]

#to acquire the skyrmion_number(topological number)
def compute_skyrmion_number(spin_map, padding='periodic'):
    # spin_map.shape must be (batch, height, width, channels) or (height, width, chennels)
    # return shape of (batch, 0) or (0,)
    solid_angle = get_solid_angle_density(spin_map, padding=padding)
    solid_angle = tf.reduce_sum(solid_angle, axis=[-3, -2])
    return solid_angle / (4. * tf.constant(np.pi, dtype=tf.float32))

#custom neural network layers
class CustomConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int = 3,
                 kernel_size: Union[int, Iterable[int]] = (3, 3),
                 strides: Union[int, Iterable[int]] = (1, 1),
                 padding: str = 'valid',
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 initializer = None,
                 **kwargs):
        super().__init__(**kwargs)
        assert padding in ['valid', 'same', 'periodic']
        if padding == 'periodic':
            self.padding = PeriodicPadding(padding=kernel_size // 2)
        else:
            self.padding = lambda x: x

        self.conv2d = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=initializer,
            padding=padding if padding != 'periodic' else 'valid'
        )
        if batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()
        else:
            self.batch_norm = lambda x: x
        if activation == 'l2norm':
            self.act = L2Normalize()
        else:
            self.act = tf.keras.layers.Activation(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.padding(inputs)
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return x

def get_callbacks(save_path, save_freq='epoch', save_best_only=False, save_weights_only=True, period=1, monitor=None,
                  patience=None, restore_best_weights=True):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                    save_freq=save_freq,
                                                    save_best_only=save_best_only,
                                                    save_weights_only=save_weights_only,
                                                    period=period)
    # earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    callbacks = [checkpoint]
    return callbacks


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, validation_data):
        super().__init__()
        self.save_dir = save_dir
        self.validation_data = validation_data
        self.losses = []
        os.makedirs(self.save_dir + "/ckpt", exist_ok=True)
        os.makedirs(self.save_dir + "/train_output", exist_ok=True)

    def on_train_begin(self, logs=None):
        # save_dir = f"models/0.0_0.0/0.0_0.0_0.0_0.0/"
        # model_dir = save_dir + str(0)
        # self.model.nn.load_weights(model_dir + "/ckpt/Ep{:07d}".format(499))
        self.model.nn.save(self.save_dir + "/model")
        self.model.nn.save_weights(self.save_dir + "/ckpt/Ep{:07d}".format(1))

    def on_train_end(self, logs=None):
        # training loss graph
        losses = np.array(self.losses)
        epochs = np.arange(len(losses))

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epochs, losses[:, 1], "k+", label='loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="k")
        ax1.tick_params(axis='y', labelcolor='k')

        ax2 = ax1.twinx()
        ax2.plot(epochs, losses[:, 2], 'rx', label='Energy Loss')
        ax2.set_ylabel('Energy Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
        ax3.plot(epochs, losses[:, 3], 'go', label='Reg Loss')
        ax3.set_ylabel('Reg Loss', color='g')
        ax3.tick_params(axis='y', labelcolor='g')

        plt.title('Train Loss')

        ax1.legend(loc='upper right')
        ax2.legend(loc='center right')
        ax3.legend(loc='lower right')

        plt.savefig(self.save_dir + "/train_loss.png")

        plt.clf()
        plt.close()


        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epochs, losses[:, 5], "k+", label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="k")
        ax1.tick_params(axis='y', labelcolor='k')

        ax2 = ax1.twinx()
        ax2.plot(epochs, losses[:, 6], 'rx', label='Energy Loss')
        ax2.set_ylabel('Energy Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax3 = ax1.twinx()
        # ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
        ax3.plot(epochs, losses[:, 7], 'go', label='Reg Loss')
        ax3.set_ylabel('Reg Loss', color='g')
        ax3.tick_params(axis='y', labelcolor='g')

        plt.title('Valid Loss')

        ax1.legend(loc='upper center')
        ax2.legend(loc='upper right')
        ax3.legend(loc='center right')

        plt.savefig(self.save_dir + "/valid_loss.png")
        plt.clf()
        plt.close()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epochs[:100], losses[:100, 5], "k+", label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="k")
        ax1.tick_params(axis='y', labelcolor='k')

        ax2 = ax1.twinx()
        ax2.plot(epochs[:100], losses[:100, 6], 'rx', label='Energy Loss')
        ax2.set_ylabel('Energy Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax3 = ax1.twinx()
        # ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
        ax3.plot(epochs[:100], losses[:100, 7], 'go', label='Reg Loss')
        ax3.set_ylabel('Reg Loss', color='g')
        ax3.tick_params(axis='y', labelcolor='g')

        plt.title('Valid Loss')

        ax1.legend(loc='upper center')
        ax2.legend(loc='upper right')
        ax3.legend(loc='center right')

        plt.savefig(self.save_dir + "/early_valid_loss.png")
        plt.clf()
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        # logs record
        self.losses.append(list(logs.values()))
        x_valid, y_valid = self.validation_data

        # image save
        val_outputs = self.model.nn.predict(x_valid, verbose=0)
        val_outputs = tf.transpose(val_outputs, [1, 0, 2, 3])
        val_outputs = tf.reshape(val_outputs, [val_outputs.shape[0], -1, val_outputs.shape[-1]])
        plt.imsave(self.save_dir + "/train_output/Ep{:07d}.png".format(epoch), spin_to_rgb(val_outputs), origin='lower')

        # ckpt save
        self.model.nn.save_weights(self.save_dir + "/ckpt/Ep{:07d}".format(epoch))


def get_rc_loss(y, y_pred):
    return K.mean(tf.sqrt(tf.square(y - y_pred)))


def wrap_pad(tfX, padi=1, padj=1, periodic="Periodic"):
    tfX = tf.concat([tfX[..., -padj:, :], tfX, tfX[..., 0:padj, :]], 2)
    tfX = tf.concat([tfX[..., -padi:, :, :], tfX, tfX[..., 0:padi, :, :]], 1)
    return tfX


def pad_spin(spin, pad=None, value=None):
    if value is None:
        value = [0., 0., 1.]
    if pad is None:
        pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
    spinx, spiny, spinz = spin[..., :1], spin[..., 1:2], spin[..., 2:]
    spinx = tf.pad(spinx, pad, constant_values=value[0])
    spiny = tf.pad(spiny, pad, constant_values=value[1])
    spinz = tf.pad(spinz, pad, constant_values=value[2])
    spin = tf.concat([spinx, spiny, spinz], axis=-1)
    return spin

#get Hamiltonian parameters
def get_heff(spins, exJ=1.0, DMN=0.3, Hextz=0.0, Kz=0.0, padding='periodic'):
    if padding == 'periodic':
        padded = wrap_pad(spins, 1, 1)
    elif padding == 'valid':
        padded = tf.concat([
            spins[..., :1, :, :], spins, spins[:, -1:, :, :]
        ], axis=-3)
        padded = tf.concat([
            padded[..., :, :1, :], padded, padded[:, :, -1:, :]
        ], axis=-2)
    heff_self = spins * 0.
    heff_exJ = tf.stack([
        padded[..., :-2, 1:-1, 0] + padded[..., 2:, 1:-1, 0] + padded[..., 1:-1, :-2, 0] + padded[..., 1:-1, 2:, 0],
        padded[..., :-2, 1:-1, 1] + padded[..., 2:, 1:-1, 1] + padded[..., 1:-1, :-2, 1] + padded[..., 1:-1, 2:, 1],
        padded[..., :-2, 1:-1, 2] + padded[..., 2:, 1:-1, 2] + padded[..., 1:-1, :-2, 2] + padded[..., 1:-1, 2:, 2]
    ], axis=-1) * exJ
    heff_DMN = tf.stack([
        -padded[..., 1:-1, 2:, 2] + padded[..., 1:-1, :-2, 2],
        -padded[..., 2:, 1:-1, 2] + padded[..., :-2, 1:-1, 2],
        padded[..., 1:-1, 2:, 0] - padded[..., 1:-1, :-2, 0] + padded[..., 2:, 1:-1, 1] - padded[..., :-2, 1:-1, 1]
    ], axis=-1) * DMN
    heff_Kz = tf.stack([
        tf.zeros_like(spins[..., 0]),
        tf.zeros_like(spins[..., 1]),
        spins[..., -1]
    ], axis=-1) * 2 * Kz
    heff_Hextz = tf.ones_like(spins) * [0.0, 0.0, Hextz]
    return (heff_self + heff_exJ + heff_DMN + heff_Kz) / 2. + heff_Hextz


def get_energy_density_map(spin, exJ=0.0, DMN=0.0, Hextz=0.0, Kz=0.0, padding='periodic'):
    heff = get_heff(spin, exJ=exJ, DMN=DMN, Hextz=Hextz, Kz=Kz, padding=padding)
    e_map = -tf.reduce_sum(spin * heff, axis=-1)
    return e_map


def get_energy(spin, exJ=1.0, DMN=0.0, Hextz=0.0, Kz=0.0, padding='periodic'):
    e_map = get_energy_density_map(spin, exJ=exJ, DMN=DMN, Hextz=Hextz, Kz=Kz, padding=padding)
    return tf.reduce_mean(e_map, axis=(1, 2))

#Custom cnn_model
class cnn_model(tf.keras.models.Model):
    def __init__(self,
                 noise_level: float = 0.0,
                 padding: str = 'periodic',
                 activation: str = 'tanh',
                 batch_norm: bool = False,
                 last_act: str = 'linear',
                 alpha: float = 0,
                 beta: float = 0,
                 exJ: float = 0,
                 DMN: float = 0,
                 Hextz: float = 0.,
                 Kz: float = 0,
                 filters: int = 32,
                 kernel_size: int = 3,
                 *args,
                 **kwargs):
        super().__init__()
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Input((None, None, 1)),
            CustomConv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                batch_norm=batch_norm,
                initializer=None
            ),
            CustomConv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                activation= activation,
                batch_norm=batch_norm,
                initializer=None
            ),
            CustomConv2D(
                filters=3,
                kernel_size=kernel_size,
                padding=padding,
                activation=last_act,
                batch_norm=batch_norm,
                initializer=None
            )
        ])
        self.alpha = alpha
        self.beta = beta
        self.exJ = exJ
        self.DMN = DMN
        self.Hextz = Hextz
        self.Kz = Kz
        self.noise_level = noise_level
                   
    def train_step(self, data):
        x, y = data
        # x = resize(contrast_layer(x))
        with tf.GradientTape() as tape:
            # x = tf.cast(x, tf.float32)
            x = tf.cast(x, tf.float32) + tf.random.normal(tf.shape(x), stddev=self.noise_level)
            spin = self.nn(x, training=True)
            # spin = tf.math.l2_normalize(spin, axis=-0)
            # spin = self(x)
            y_pred = compute_skyrmion_number(spin, padding='periodic')
            loss = self.compiled_loss(y, y_pred)
            energy_loss = tf.reduce_mean(
                get_energy(tf.math.l2_normalize(spin, axis=-1), exJ=self.exJ, DMN=self.DMN, Hextz=self.Hextz,
                           Kz=self.Kz, padding='periodic'))
            norm_outputs = tf.reduce_sum(tf.square(spin), axis=-1)
            reg_loss = tf.square((tf.ones_like(norm_outputs) - norm_outputs))
            reg_loss = tf.reduce_mean(reg_loss)
            total_loss = loss + self.alpha * energy_loss + self.beta * reg_loss
            # tf.print(norm_outputs)

            # # accuracy metric
            # normalized_spin = tf.math.l2_normalize(spin, axis=-0)
            # y_pred_normed = compute_skyrmion_number(normalized_spin, padding='periodic')
            # error_normed = y_pred_normed - y
            # correct = tf.math.clip

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {"total_loss": total_loss, "loss": loss, "energy_loss": energy_loss, "reg_loss": reg_loss}

    def test_step(self, data):
        x, y = data
        # x = resize(contrast_layer(x))
        x = x + tf.random.normal(tf.shape(x), stddev=self.noise_level),
        spin = self.nn(x, training=False)
        # spin = self(x)
        y_pred = compute_skyrmion_number(spin, padding='periodic')
        loss = self.compiled_loss(y, y_pred)
        energy_loss = tf.reduce_mean(
            get_energy(tf.math.l2_normalize(spin, axis=-1), exJ=self.exJ, DMN=self.DMN, Hextz=self.Hextz, Kz=self.Kz,
                       padding='periodic'))
        norm_outputs = tf.reduce_sum(tf.square(spin), axis=-1)
        reg_loss = tf.square((tf.ones_like(norm_outputs) - norm_outputs))
        reg_loss = tf.reduce_mean(reg_loss)

        # tf.print(tf.shape(reg_loss))
        # tf.print(tf.shape(energy_loss))
        total_loss = loss + self.alpha * energy_loss + self.beta * reg_loss
        self.compiled_metrics.update_state(y, y_pred)
        return {"total_loss": total_loss, "loss": loss, "energy_loss": energy_loss, "reg_loss": reg_loss}


