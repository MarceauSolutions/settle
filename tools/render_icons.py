#!/usr/bin/env python3
"""Draw the home-screen icons into icons/. Needs Pillow."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "icons"
SCALE = 1024
BG = (24, 22, 20)
GOLD = (201, 150, 60)


def draw():
    image = Image.new("RGB", (SCALE, SCALE), BG)
    glow = Image.new("RGB", (SCALE, SCALE), BG)
    centre = SCALE // 2
    ImageDraw.Draw(glow).ellipse([centre - 300, centre - 300, centre + 300, centre + 300], fill=(92, 70, 32))
    image.paste(glow.filter(ImageFilter.GaussianBlur(90)))
    pen = ImageDraw.Draw(image)
    pen.ellipse([centre - 205, centre - 205, centre + 205, centre + 205], outline=GOLD, width=26)
    pen.ellipse([centre - 80, centre - 80, centre + 80, centre + 80], fill=GOLD)
    return image


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    icon = draw()
    for name, size in (("apple-touch-icon", 180), ("icon-192", 192), ("icon-512", 512)):
        icon.resize((size, size), Image.LANCZOS).save(OUT / f"{name}.png")
