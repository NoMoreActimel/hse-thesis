from PIL import Image, ImageDraw, ImageFont
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import os
import textwrap

from core.utils.example_utils import hstack_with_lines, vstack_with_lines

def stack_rows(rows, skip_horiz=10, skip_vertical=15):
    final_image = [
        hstack_with_lines(row_stack, skip_horiz) for row_stack in rows
    ]

    final_image = vstack_with_lines(final_image, skip_vertical)
    return Image.fromarray(final_image)

def column_named(grid, names, font=ImageFont.truetype("/home/sasedov/Times.ttf", 25), H=1024, W=1024, alpha=1, text_field_height=50):
    draw = ImageDraw.Draw(grid)
    W_, H_ = grid.size
    ws, hw = zip(*[draw.textbbox((0, 0), name, font=font)[-2:] for name in names])
    
    new_img = Image.new('RGB', (W_, H_ + max(hw) + text_field_height), (255, 255, 255))
    if alpha != 1:
        alpha = Image.new("L", new_img.size, int(255 * alpha))
        new_img.putalpha(alpha)
    
    draw = ImageDraw.Draw(new_img)
    W_, H_ = new_img.size
    new_img.paste(grid, box=(0, max(hw) + text_field_height))
    for i, name in enumerate(names):
        draw.text((W * i + (W-ws[i])/2, max(hw) - min(hw) + text_field_height // 2), name, font=font, fill='black')
    return new_img

def stack_to_grid_with_names(
        imgs_list, H=256, W=256,
        row_names=None, column_names=None,
        font=ImageFont.truetype("/home/sasedov/Times.ttf", 25),
        text_field_height=50, skip_horiz=10, skip_vertical=15):
    """
    imgs_list: list of lists of images, will be concatenated with skip_horiz and skip_vertical spaces
    H, W: size of each image
    """
    grid = stack_rows(imgs_list, skip_horiz=skip_horiz, skip_vertical=skip_vertical)

    if column_names is not None:
        grid = column_named(
            grid=grid, names=column_names, font=font,
            H=H, W=W + skip_horiz, text_field_height=text_field_height
        )
        if row_names is not None:
            W = (grid.size[1] - text_field_height) // len(row_names)
    
    if row_names is not None:
        grid = column_named(
            grid.rotate(-90, expand=True), names=row_names[::-1], font=font,
            H=H, W=W, text_field_height=text_field_height
        ).rotate(90, expand=True)

    return grid


def text_on_square_image(text, image_size, font=ImageFont.truetype("/home/sasedov/Times.ttf", 25), linewidth=20):
    im = Image.new('RGB', (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    y_text = image_size / 2
    lines = textwrap.wrap(text, width=linewidth)
    for line in lines:
        width, height = font.getbbox(line)[2:]
        print(width, height)
        draw.text(((image_size - width) / 2, y_text), line, font=font, fill='black')
        y_text += height

    # draw.text(((image_size - w) / 2, (image_size - h) / 2), text, font=font, fill='black')
    return np.array(im)
