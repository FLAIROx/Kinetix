import os
import pathlib

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
from PIL import Image


def load_texture(filename, render_size):
    filename = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "assets", filename)
    img = iio.imread(filename)
    jnp_img = jnp.array(img).astype(jnp.int32)

    if jnp_img.shape[2] == 4:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    img = np.array(jnp_img, dtype=np.uint8)
    image = Image.fromarray(img)
    image = image.resize((render_size, render_size), resample=Image.NEAREST)
    jnp_img = jnp.array(image, dtype=jnp.float32)

    return jnp_img.transpose((1, 0, 2))


EDIT_TEXTURE_RGBA = load_texture("edit.png", 64)
PLAY_TEXTURE_RGBA = load_texture("play.png", 64)

CIRCLE_TEXTURE_RGBA = load_texture("circle.png", 32)
RECT_TEXTURE_RGBA = load_texture("square.png", 32)
TRIANGLE_TEXTURE_RGBA = load_texture("triangle.png", 32)
RJOINT_TEXTURE_6_RGBA = load_texture("rjoint.png", 6)
RJOINT_TEXTURE_RGBA = load_texture("rjoint2.png", 32)

FJOINT_TEXTURE_6_RGBA = load_texture("fjoint.png", 6)
FJOINT_TEXTURE_RGBA = load_texture("fjoint2.png", 32)


ROTATION_TEXTURE_RGBA = load_texture("rotate.png", 32)
SELECT_TEXTURE_RGBA = load_texture("hand.png", 32)

THRUSTER_TEXTURE_RGBA = jnp.rot90(load_texture("thruster6.png", 32), k=3)
THRUSTER_TEXTURE_16_RGBA = jnp.rot90(load_texture("thruster.png", 16), k=3)
