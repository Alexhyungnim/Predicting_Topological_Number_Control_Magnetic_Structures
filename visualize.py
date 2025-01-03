from mayavi import mlab
from tvtk.api import tvtk
import numpy as np
from typing import Union
from matplotlib import pyplot as plt
from Custom_Modules.modules import spin_to_rgb, compute_skyrmion_number, get_energy
from Custom_Modules.modules import AddNoise, CustomConv2D
import tensorflow as tf

#parameters
ALPHA, BETA = 0.1, 1.0  # Hamiltonian, Regularization
EXJ, DMN, HEXTZ, KZ = 1.0, 0.5, 0.0, 0.05
EPOCH = 1999
BATCH_SIZE = 1
MODNUM = 1


# Predict spin configuration
x_test = np.load("Data/ordinary_train/circle.npy").astype(np.float32)[0:1]
save_dir = f"models/{ALPHA}_{BETA}/{EXJ}_{DMN}_{HEXTZ}_{KZ}/"
model_dir = save_dir + str(MODNUM)
model = tf.keras.models.load_model(model_dir + "/model")
model.load_weights(model_dir + "/ckpt/Ep{:07d}".format(EPOCH))
spin = model(x_test[0:1])
plt.imshow(spin_to_rgb(spin[0]))
plt.show()

sz_profile = tf.math.l2_normalize(spin, axis=-1)[0, 64, 80:112, -1]

# plot the Spin_z component, and compare this(experimental) to theoretical values
def plot_sz_profile(sz_profile: Union[np.ndarray, tf.Tensor], Kz: float):
    # Interpolate sz_profile to find where sz value is 0
    indices = np.arange(len(sz_profile))
    new_indices = np.linspace(indices[0], indices[-1], 1000)
    new_sz_profile = np.interp(new_indices, indices, sz_profile)

    # Find where sz is closest to 0
    zero_crossing_index = np.argmin(np.abs(new_sz_profile))
    zero_position = new_indices[zero_crossing_index]

    # Plot sz_profile with zero crossing centered and reference line at Kz
    fig, ax = plt.subplots()
    ax.plot(indices - zero_position, sz_profile, 'k+', label='$s_z$ Profile')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    x_ref = np.linspace(new_indices[0] - zero_position, new_indices[-1] - zero_position, 1000)
    # Decide sign of tanh to match sz_profile
    tanh_ref_pos = np.tanh(x_ref / np.sqrt(0.5 / Kz))
    tanh_ref_neg = -tanh_ref_pos
    diff_pos = np.sum(np.abs(new_sz_profile - tanh_ref_pos))
    diff_neg = np.sum(np.abs(new_sz_profile - tanh_ref_neg))
    tanh_ref = tanh_ref_pos if diff_pos < diff_neg else tanh_ref_neg
    ax.plot(x_ref, tanh_ref, color='red', linestyle='--', linewidth=1, label=f'Reference tanh(Kz={Kz})')
    ax.set_xlim(-10, 10)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # ax.set_xlabel('Position (shifted)')
    # ax.set_ylabel('$s_z$ Value')
    # ax.set_title('$s_z$ Profile with Zero Crossing Centered')
    ax.legend(frameon = True, fontsize=12, loc = 'upper right')
    plt.savefig(f'Kz{KZ}.png')
    plt.show()

plot_sz_profile(sz_profile, Kz=KZ)
# Map the whole spin vectors in 3D visualization 
def plot_vectors(vectors: Union[np.ndarray, tf.Tensor]):
    """
    Plot 3D vectors using Mayavi.

    Parameters:
    vectors: Union[np.ndarray, tf.Tensor]
        A tensor or array of shape [N, 3] representing N vectors in 3D space.
    """
    # Set background color to white
    mlab.figure(bgcolor=(1, 1, 1))
    # Extract x, y, z coordinates
    x, y, z = vectors[:, 0].numpy(), vectors[:, 1].numpy(), vectors[:, 2].numpy()

    # Generate colors for the vectors
    colors_list = spin_to_rgb(vectors)
    colors_list = np.clip(colors_list, 0., 1.) * 255  # Convert to range [0, 255]
    colors_list = colors_list.astype(np.uint8)
    colors_list = np.concatenate([colors_list, np.ones_like(colors_list[..., :1]) * 255], axis=-1)
    colors_list[:2] = [255, 0, 0, 255]  # Set specific colors for first two points

    # Plot the points
    pts = mlab.pipeline.scalar_scatter(x, y, z)
    pts.add_attribute(colors_list, 'colors')
    pts.data.point_data.set_active_scalars('colors')
    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = 0.05
    g.glyph.scale_mode = 'data_scaling_off'

    # Add sphere with radius 0.95 to occlude points behind
    phi, theta = np.mgrid[0:np.pi:100j, 0:2 * np.pi:100j]
    x_sphere = 0.95 * np.sin(phi) * np.cos(theta)
    y_sphere = 0.95 * np.sin(phi) * np.sin(theta)
    z_sphere = 0.95 * np.cos(phi)
    mlab.mesh(x_sphere, y_sphere, z_sphere, color=(0.5, 0.5, 0.5), opacity=0.4)

    # Add an orientation axes
    orientation_axes = mlab.orientation_axes()

    # Setup color for each axis
    orientation_axes.axes.normalized_label_position = [1.5, 1.5, 1.5]
    orientation_axes.axes.x_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
    orientation_axes.axes.y_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)
    orientation_axes.axes.z_axis_caption_actor2d.caption_text_property.color = (0, 0, 0)

    # Set camera view
    mlab.view(azimuth=-0.4 * 180, elevation=0.4 * 180)

    # Display the plot
    mlab.show()

vectors = tf.reshape(tf.math.l2_normalize(spin, axis=-1), (-1, 3))
vectors = tf.reshape(tf.math.l2_normalize(spin[0, ::2, ::2], axis=-1), (-1, 3))

# Call the function to plot
plot_vectors(vectors)
