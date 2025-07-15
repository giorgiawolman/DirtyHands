import cv2
import os

def split_image(image_path, output_dir, tile_size=64, pad=True, pad_value=128):
    """
    Splits the input image into tiles of size tile_size × tile_size.
    If pad is True, pads the image so its dimensions are multiples of tile_size.
    Saves each tile as a separate PNG in output_dir.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return

    h, w = img.shape[:2]

    # Apply padding if requested
    if pad:
        pad_h = (tile_size - (h % tile_size)) % tile_size
        pad_w = (tile_size - (w % tile_size)) % tile_size
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        # Use reflective padding to avoid hard edges
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        h, w = img.shape[:2]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    # Slide window over image
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            # Skip incomplete tiles if pad is False
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(output_dir, f"{base}_{y}_{x}.png")
            cv2.imwrite(out_path, tile)
            count += 1

    print(f"Saved {count} tiles to {output_dir}")

# === Configuration: Hard-coded paths ===
INPUT_IMAGE = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\Images\HighRes TrueColor 2018 2m Graz.png"
OUTPUT_DIR  = r"C:\Users\Lennart Hamm\Desktop\divers\macad\thesis\DirtyHands\Images\ImageSplit"

if __name__ == "__main__":
    # Split the image into 64×64 tiles with padding enabled
    split_image(
        image_path=INPUT_IMAGE,
        output_dir=OUTPUT_DIR,
        tile_size=64,
        pad=True
    )
