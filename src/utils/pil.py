from PIL import Image, ImageDraw


def open_image(path, inplane=None):
    image = Image.open(path)
    if inplane is not None:
        image = image.rotate(inplane)
    return image


def draw_box(img_PIL, box, color="red", width=3):
    draw = ImageDraw.Draw(img_PIL)
    draw.rectangle(
        (
            (box[0], box[1]),
            (box[2], box[3]),
        ),
        outline=color,
        width=width,
    )
    return img_PIL


def draw_points(img_PIL, points_2d):
    draw = ImageDraw.Draw(img_PIL)
    for point in points_2d:
        draw.rectangle(
            ((point[0] - 0.1, point[1] + 0.1), (point[0] - 0.1, point[1] + 0.1)),
            outline="blue",
            width=5,
        )
    return img_PIL
