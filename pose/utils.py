# utils.py

from PIL import Image, ImageDraw

def make_circular(image_path):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((200, 200))
    bigsize = (img.size[0] * 3, img.size[1] * 3)
    mask = Image.new("L", bigsize, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + bigsize, fill=255)
    mask = mask.resize(img.size, Image.ANTIALIAS)
    img.putalpha(mask)
    return img
