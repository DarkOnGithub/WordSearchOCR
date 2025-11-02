"""
Font-Based Letter Dataset Builder with Advanced Augmentations

Creates a dataset of 28x28 letter images with strong augmentations for OCR training.
Features:
- Generates letters a-z (upper/lower) from common printing fonts
- Custom combined crumple+print effects using grid processing technique
- Additional augmentations using Albumentations
- Parallel processing for efficiency
- IDX format output compatible with C dataloader
- ~100K images total (85/15 train/test split)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations import Compose
from albumentations.core.transforms_interface import ImageOnlyTransform
import struct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
from PIL import Image, ImageDraw, ImageFont
import string

def create_low_freq_noise(width, height, low_res_w, low_res_h):
    """
    Create low frequency noise for crumple effects.
    """
    small_noise_map = np.random.rand(low_res_h, low_res_w) * 2 - 1
    noise_map = cv2.resize(small_noise_map, (width, height), interpolation=cv2.INTER_CUBIC)
    return noise_map

class CombinedCrumplePrintTransform(ImageOnlyTransform):
    """
    Custom transform combining crumple and print effects.
    Applies effects to a larger canvas for more realistic results.
    Enhanced with printing artifacts and paper aging effects.
    """

    def __init__(self,
                 crumple_strength=15,
                 crease_intensity=160,
                 crumple_scale_w=30,
                 crumple_scale_h=22,
                 blur_kernel=(5, 5),
                 contrast_alpha=0.78,
                 brightness_beta=35,
                 noise_scale=5,
                 canvas_scale=3.5,
                 ink_bleed_strength=0.3,
                 paper_yellowing=0.15,
                 always_apply=False,
                 p=1.0):
        super(CombinedCrumplePrintTransform, self).__init__(always_apply, p)
        self.crumple_strength = crumple_strength
        self.crease_intensity = crease_intensity
        self.crumple_scale_w = crumple_scale_w
        self.crumple_scale_h = crumple_scale_h
        self.blur_kernel = blur_kernel
        self.contrast_alpha = contrast_alpha
        self.brightness_beta = brightness_beta
        self.noise_scale = noise_scale
        self.canvas_scale = canvas_scale
        self.ink_bleed_strength = ink_bleed_strength
        self.paper_yellowing = paper_yellowing

    def apply(self, image, **params):
        height, width = image.shape[:2]

        # Create larger canvas
        canvas_height = int(height * self.canvas_scale)
        canvas_width = int(width * self.canvas_scale)

        # Create canvas with light background (white bg style)
        canvas = np.random.normal(240, 10, (canvas_height, canvas_width)).astype(np.uint8)
        if len(image.shape) == 3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Randomly place the original image on the canvas
        max_offset_h = canvas_height - height
        max_offset_w = canvas_width - width

        if max_offset_h > 0 and max_offset_w > 0:
            offset_h = np.random.randint(0, max_offset_h)
            offset_w = np.random.randint(0, max_offset_w)
        else:
            offset_h = offset_w = 0

        canvas[offset_h:offset_h+height, offset_w:offset_w+width] = image

        # === 1. Apply print effects first ===
        processed = cv2.blur(canvas, self.blur_kernel)
        processed = cv2.convertScaleAbs(processed, alpha=self.contrast_alpha, beta=self.brightness_beta)

        # === 1.5. Add ink bleeding effect ===
        if self.ink_bleed_strength > 0:
            # Create ink bleeding by dilating dark areas slightly
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            ink_bleed = cv2.dilate(processed, kernel, iterations=1)
            # Blend with original for subtle effect
            processed = cv2.addWeighted(processed, 1-self.ink_bleed_strength, ink_bleed, self.ink_bleed_strength, 0)

        # === 1.6. Add paper yellowing effect ===
        if self.paper_yellowing > 0:
            # Create subtle yellow tint
            yellow_overlay = np.full_like(processed, [240, 235, 220], dtype=np.uint8)  # Slightly yellow paper color
            if len(processed.shape) == 3:
                processed = cv2.addWeighted(processed, 1-self.paper_yellowing, yellow_overlay, self.paper_yellowing, 0)
            else:
                # For grayscale, add slight darkening/yellowing
                yellow_factor = 1 - self.paper_yellowing * 0.1
                processed = cv2.convertScaleAbs(processed, alpha=yellow_factor, beta=self.paper_yellowing * 10)

        # === 2. Apply crumple effects ===
        # Create distortion maps
        noise_x = create_low_freq_noise(canvas_width, canvas_height, self.crumple_scale_w, self.crumple_scale_h)
        noise_y = create_low_freq_noise(canvas_width, canvas_height, self.crumple_scale_w, self.crumple_scale_h)

        # Create zone mask
        zone_map = create_low_freq_noise(canvas_width, canvas_height, 6, 5)
        zone_mask = cv2.normalize(zone_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
        zone_mask = np.power(zone_mask, 2)
        zone_mask = cv2.GaussianBlur(zone_mask, (81, 81), 0)

        # Calculate geometric distortion
        displacement_x = noise_x * self.crumple_strength * zone_mask
        displacement_y = noise_y * self.crumple_strength * zone_mask

        x_coords, y_coords = np.meshgrid(np.arange(canvas_width), np.arange(canvas_height))
        map_x = (x_coords + displacement_x).astype(np.float32)
        map_y = (y_coords + displacement_y).astype(np.float32)

        warped_canvas = cv2.remap(processed, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Create crease lighting
        noise_y_32f = noise_y.astype(np.float32)
        grad_x = cv2.Sobel(noise_y_32f, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(noise_y_32f, cv2.CV_32F, 0, 1, ksize=5)
        crease_map = np.abs(grad_x) + np.abs(grad_y)

        shading_map = cv2.normalize(crease_map, None, 0, 1.0, cv2.NORM_MINMAX)
        shading_map = shading_map * self.crease_intensity * zone_mask

        # Apply crease lighting
        warped_with_shading = warped_canvas.astype(np.int32) + shading_map.astype(np.int32)
        warped_with_shading = np.clip(warped_with_shading, 0, 255).astype(np.uint8)

        # === 3. Add final paper grain texture ===
        grain_noise = np.random.normal(loc=0, scale=self.noise_scale, size=warped_with_shading.shape).astype(np.int16)
        final_canvas = np.clip(warped_with_shading.astype(np.int16) + grain_noise, 0, 255).astype(np.uint8)

        # Extract the region where the original image was placed
        result = final_canvas[offset_h:offset_h+height, offset_w:offset_w+width]

        return result

def get_common_fonts():
    """
    Get list of common printing fonts available on most systems.
    Returns fonts that are readable and not artistic.
    """
    # Comprehensive list of common system fonts (Windows/Linux/Mac compatible)
    font_candidates = [
        # Arial family
        "arial.ttf", "Arial.ttf", "ARIAL.TTF",
        "arialbd.ttf", "Arialbd.ttf", "ARIALBD.TTF",
        "arialbi.ttf", "Arialbi.ttf", "ARIALBI.TTF",
        "ariali.ttf", "Ariali.ttf", "ARIALI.TTF",

        # Times New Roman family
        "times.ttf", "Times.ttf", "TIMES.TTF",
        "timesbd.ttf", "Timesbd.ttf", "TIMESBD.TTF",
        "timesbi.ttf", "Timesbi.ttf", "TIMESBI.TTF",
        "timesi.ttf", "Timesi.ttf", "TIMESI.TTF",

        # Courier family
        "cour.ttf", "Cour.ttf", "COUR.TTF",
        "courbd.ttf", "Courbd.ttf", "COURBD.TTF",
        "courbi.ttf", "Courbi.ttf", "COURBI.TTF",
        "couri.ttf", "Couri.ttf", "COURI.TTF",

        # Calibri family
        "calibri.ttf", "Calibri.ttf", "CALIBRI.TTF",
        "calibrib.ttf", "Calibrib.ttf", "CALIBRIB.TTF",
        "calibrii.ttf", "Calibrii.ttf", "CALIBRII.TTF",
        "calibril.ttf", "Calibril.ttf", "CALIBRIL.TTF",

        # Tahoma family
        "tahoma.ttf", "Tahoma.ttf", "TAHOMA.TTF",
        "tahomabd.ttf", "Tahomabd.ttf", "TAHOMABD.TTF",

        # Verdana family
        "verdana.ttf", "Verdana.ttf", "VERDANA.TTF",
        "verdanab.ttf", "Verdanab.ttf", "VERDANAB.TTF",
        "verdanai.ttf", "Verdanai.ttf", "VERDANAI.TTF",
        "verdanaz.ttf", "Verdanaz.ttf", "VERDANAZ.TTF",

        # Georgia family
        "georgia.ttf", "Georgia.ttf", "GEORGIA.TTF",
        "georgiab.ttf", "Georgiab.ttf", "GEORGIAB.TTF",
        "georgiai.ttf", "Georgiai.ttf", "GEORGIAI.TTF",
        "georgiaz.ttf", "Georgiaz.ttf", "GEORGIAZ.TTF",

        # Helvetica (common on Mac/Linux)
        "Helvetica.ttf", "helvetica.ttf", "HELVETICA.TTF",
        "Helvetica-Bold.ttf", "helvetica-bold.ttf",

        # Liberation fonts (common on Linux)
        "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
        "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
        "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf",

        # DejaVu fonts (common on Linux)
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
        "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
        "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf",

        # Ubuntu fonts (common on Ubuntu)
        "Ubuntu-Regular.ttf", "Ubuntu-Bold.ttf",
        "UbuntuMono-Regular.ttf", "UbuntuMono-Bold.ttf",

        # Segoe UI (Windows 7+)
        "segoeui.ttf", "Segoeui.ttf", "SEGOEUI.TTF",
        "segoeuib.ttf", "Segoeuib.ttf", "SEGOEUIB.TTF",
        "segoeuil.ttf", "Segoeuil.ttf", "SEGOEUIL.TTF",

        # Cambria (Windows Office)
        "cambria.ttc", "Cambria.ttc", "CAMBRIATTC",
        "cambriab.ttf", "Cambriab.ttf", "CAMBRIAB.TTF",

        # Consolas (Windows programming font)
        "consola.ttf", "Consola.ttf", "CONSOLA.TTF",
        "consolab.ttf", "Consolab.ttf", "CONSOLAB.TTF",

        # Lucida Console
        "lucon.ttf", "Lucon.ttf", "LUCON.TTF",

        # System fonts (fallback)
        "system.ttf", "System.ttf", "SYSTEM.TTF",
        "sfns.ttf", "Sfns.ttf", "SFNS.TTF",  # San Francisco
    ]

    # Comprehensive font directories for different systems
    font_dirs = [
        # Windows
        "C:/Windows/Fonts/",
        "C:/WINNT/Fonts/",

        # macOS
        "/System/Library/Fonts/",
        "/Library/Fonts/",
        "/System/Library/Assets/com_apple_MobileAsset_Font5/",
        "~/Library/Fonts/",

        # Linux common directories
        "/usr/share/fonts/",
        "/usr/share/fonts/truetype/",
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
        "/usr/share/fonts/truetype/ubuntu/",
        "/usr/share/fonts/truetype/freefont/",
        "/usr/share/fonts/TTF/",
        "/usr/share/fonts/OTF/",
        "/usr/local/share/fonts/",

        # Additional Linux paths
        "/usr/X11R6/lib/X11/fonts/TTF/",
        "/usr/X11R6/lib/X11/fonts/truetype/",
        "/var/lib/defoma/x-ttcidfont-conf.d/dirs/TrueType/",

        # BSD systems
        "/usr/local/share/fonts/",

        # Android (if running on Android)
        "/system/fonts/",

        # Current directory and subdirectories
        "./fonts/",
        "../fonts/",
        "../../fonts/",
    ]

    available_fonts = []

    # Search for fonts in all directories
    for font_dir in font_dirs:
        try:
            expanded_dir = os.path.expanduser(font_dir)
            if os.path.exists(expanded_dir):
                for font_file in font_candidates:
                    font_path = os.path.join(expanded_dir, font_file)
                    if os.path.exists(font_path):
                        available_fonts.append(font_path)
        except Exception as e:
            print(f"Warning: Could not search directory {font_dir}. Error: {e}")

    # Remove duplicates while preserving order
    seen = set()
    available_fonts = [x for x in available_fonts if not (x in seen or seen.add(x))]

    print(f"Found {len(available_fonts)} system fonts")

    # If no fonts found, try to use default PIL fonts
    if not available_fonts:
        try:
            # This will work on systems with default fonts
            ImageFont.load_default()
            available_fonts = ["default"]
            print("Using PIL default font")
        except Exception as e:
            print(f"Warning: PIL default font failed: {e}")

    # If still no fonts, create synthetic fonts as fallback
    if not available_fonts:
        print("Warning: No fonts found. Creating synthetic fonts.")
        available_fonts = ["synthetic"]

    return available_fonts

def generate_synthetic_font(letter, font_size=24, image_size=28):
    """
    Generate a synthetic letter image when no fonts are available.
    Creates a simple bitmap-style letter.
    """
    # Create image with white background
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    # Use PIL default as last resort, or create a simple synthetic version
    try:
        font = ImageFont.load_default()
        # Get text bounding box
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        x = (image_size - text_width) // 2 - bbox[0]
        y = (image_size - text_height) // 2 - bbox[1]

        # Draw the letter in black
        draw.text((x, y), letter, fill=0, font=font)
    except:
        # If even PIL default fails, create a very simple synthetic letter
        # This is a last resort fallback
        if letter.isupper():
            # Simple uppercase letter pattern (just draw a rectangle for now)
            draw.rectangle([8, 6, 20, 22], fill=0)
        else:
            # Simple lowercase letter pattern
            draw.rectangle([8, 10, 20, 22], fill=0)

    return np.array(img).astype(np.uint8)

def generate_letter_image(letter, font_path, font_size=24, image_size=28):
    """
    Generate a 28x28 image of a letter using the specified font.
    """
    # Create image with white background
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    try:
        if font_path == "default":
            font = ImageFont.load_default()
        elif font_path == "synthetic":
            # Use synthetic font generation
            return generate_synthetic_font(letter, font_size, image_size)
        else:
            font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        # Try fallback options
        try:
            font = ImageFont.load_default()
        except:
            # Last resort: synthetic font
            return generate_synthetic_font(letter, font_size, image_size)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    x = (image_size - text_width) // 2 - bbox[0]
    y = (image_size - text_height) // 2 - bbox[1]

    # Draw the letter in black
    draw.text((x, y), letter, fill=0, font=font)

    # Convert to numpy array (keep white background with black letters)
    img_array = np.array(img)

    return img_array.astype(np.uint8)

def create_base_letter_images(letters, fonts, images_per_letter=50):
    """
    Generate base letter images for each letter using different fonts and variations.
    """
    base_images = {}

    for letter in letters:
        print(f"Generating base images for letter '{letter}'...")
        letter_images = []

        for _ in range(images_per_letter):
            # Randomly select font
            font_path = random.choice(fonts)

            # Enhanced font size variation for more diversity
            font_size = random.randint(18, 32)  # Wider range: 18-32 instead of 22-26

            # Random thickness variation (if supported by font)
            # We'll add this as a parameter to generate_letter_image

            # Generate image
            img = generate_letter_image(letter, font_path, font_size)

            letter_images.append(img)

        base_images[letter] = np.array(letter_images)
        print(f"   Generated {len(letter_images)} base images for '{letter}'")

    return base_images

def create_augmentation_pipeline():
    """
    Create the main augmentation pipeline using Albumentations.
    Enhanced with sophisticated OCR-like distortions for maximum robustness.
    """
    return Compose([
        # Enhanced crumple+print effect with ink bleeding and paper aging
        CombinedCrumplePrintTransform(
            crumple_strength=15, crease_intensity=160,    # Increased for more realistic crumpling
            crumple_scale_w=30, crumple_scale_h=22,       # Larger scale for natural effects
            blur_kernel=(5, 5), contrast_alpha=0.78, brightness_beta=35,  # More pronounced printing effects
            noise_scale=5, canvas_scale=3.5,               # Larger canvas for better effects
            ink_bleed_strength=0.3, paper_yellowing=0.15,  # New OCR-specific effects
            p=0.85    # Higher probability for more variety
        ),

        # Geometric transformations - comprehensive and realistic
        A.Rotate(limit=15, p=0.7),  # Increased rotation range
        A.Perspective(scale=(0.04, 0.10), p=0.6),  # More perspective distortion
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1),
                rotate=(-5, 5), shear=(-5, 5), p=0.65),  # More comprehensive affine

        # Blur effects - multiple types for different scanning scenarios
        A.GaussNoise(std_range=(0.02, 0.15), p=0.45),    # Wider noise range
        A.GaussianBlur(blur_limit=(3, 9), p=0.3),        # Increased blur range
        A.MotionBlur(blur_limit=(3, 7), p=0.2),          # Increased motion blur
        A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=0.15),  # Camera defocus effect
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.4),  # Enhanced grain effect

        # Lighting and contrast variations - more realistic ranges
        A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15),
                                  contrast_limit=(-0.15, 0.15), p=0.5),

        # Advanced OCR distortions
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.25),  # Stronger elastic deformation
        A.ElasticTransform(alpha=2, sigma=50, p=0.2),     # Increased paper elasticity
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.15),  # Lens distortion

        # Ink and printing artifacts
        A.GaussNoise(std_range=(0.005, 0.08), p=0.4),    # Fine ink texture noise
        A.Sharpen(alpha=(0.0, 0.3), lightness=(0.8, 1.2), p=0.2),  # Ink sharpening effects
        A.UnsharpMask(alpha=(0.0, 0.2), sigma=1.0, threshold=10, p=0.15),  # Printing halftone effect

        # Scanner-specific artifacts (very subtle)
        A.GridDropout(ratio=0.3, unit_size_min=2, unit_size_max=4,
                     shifts=True, random_offset=True, fill_value=255, p=0.1),  # Scanner dust/specks
        A.CoarseDropout(max_holes=8, max_height=2, max_width=2,
                       min_holes=1, min_height=1, min_width=1, fill_value=255, p=0.1),  # Small imperfections
    ])

def apply_augmentations(image, pipeline, num_augmentations=8):
    """
    Apply augmentations to a single image.
    """
    results = [image]  # Include original

    for _ in range(num_augmentations):
        augmented = pipeline(image=image)
        aug_img = augmented['image']
        # Ensure it's still 28x28
        if aug_img.shape != (28, 28):
            aug_img = cv2.resize(aug_img, (28, 28))
        results.append(aug_img.astype(np.uint8))

    return results

def process_letter_batch(letter, images, label, pipeline, augmentations_per_image):
    """
    Process a batch of images for a single letter.
    """
    all_images = []
    all_labels = []

    for img in images:
        augmented_images = apply_augmentations(img, pipeline, augmentations_per_image)
        all_images.extend(augmented_images)
        all_labels.extend([label] * len(augmented_images))

    return all_images, all_labels

def save_idx_images(filename, images):
    """Save images in IDX format"""
    num_images, height, width = images.shape
    magic = 2051
    header = struct.pack('>IIII', magic, num_images, height, width)

    with open(filename, 'wb') as f:
        f.write(header)
        for img in images:
            f.write(img.tobytes())

def save_idx_labels(filename, labels):
    """Save labels in IDX format"""
    num_labels = len(labels)
    magic = 2049
    header = struct.pack('>II', magic, num_labels)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(labels.astype(np.uint8).tobytes())

def create_font_based_dataset(target_dir, total_images=100000, train_split=0.85, save_samples=False):
    """
    Create the complete dataset.

    Args:
        save_samples: If True, saves sample images for each letter for visualization
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True) # [--- FIX ---] Added parents=True

    # Define letters (a-z upper and lower, but same labels for upper/lower case)
    letters = string.ascii_letters  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # Map both upper and lower case to same label (a=1, A=1, b=2, B=2, etc.)
    letter_to_label = {letter: ord(letter.lower()) - ord('a') for letter in letters} # Using 1-based indexing

    unique_classes = len(set(letter_to_label.values()))
    print(f"Creating dataset with {unique_classes} letter classes (upper/lower case share labels)")
    print(f"Processing {len(letters)} letter variants: {letters}")
    print(f"Target: {total_images} total images ({train_split*100:.0f}% train, {(1-train_split)*100:.0f}% test)")

    # Get available fonts
    fonts = get_common_fonts()
    if not fonts or fonts == ["synthetic"]:
         print("CRITICAL WARNING: No real fonts found. Dataset will be low quality.")
    print(f"Using {len(fonts)} available fonts")

    # Calculate distribution
    images_per_letter = total_images // len(letters)  # ~1923 per letter variant (e.g., 'a' and 'A' are separate)
    base_images_per_letter = max(50, images_per_letter // 40) # [--- FIX ---] Increased base images

    print(f"Generating {base_images_per_letter} base images per letter variant, then augmenting to ~{images_per_letter} total per variant")

    # Generate base images
    print("\nGenerating base letter images...")
    base_images = create_base_letter_images(letters, fonts, base_images_per_letter)

    # Create augmentation pipeline
    pipeline = create_augmentation_pipeline()

    # Calculate augmentations per base image
    # We add 1 to base_images_per_letter to avoid division by zero if it's 0
    # The -1 is because process_letter_batch includes the original
    augmentations_per_image = max(0, (images_per_letter // (base_images_per_letter + 1)) - 1)

    print(f"\nSplitting base images and augmenting ({augmentations_per_image} augmentations per base train image)...")

    # Process each letter - split BEFORE augmenting
    all_train_images = []
    all_train_labels = []
    all_test_images = []
    all_test_labels = []

    for letter in letters:
        print(f"\nProcessing letter '{letter}' (label {letter_to_label[letter]})...")

        base_letter_images = base_images[letter]
        label = letter_to_label[letter]

        # Shuffle base images first
        indices = list(range(len(base_letter_images)))
        random.shuffle(indices)
        base_letter_images = base_letter_images[indices]

        # Split base images into train/test BEFORE augmentation
        split_idx = int(len(base_letter_images) * train_split)
        train_base_images = base_letter_images[:split_idx]
        test_base_images = base_letter_images[split_idx:]

        print(f"   Base images: {len(train_base_images)} train, {len(test_base_images)} test")

        # [--- FIX: THIS IS THE CORE LOGIC CHANGE ---]

        # 1. Augment train base images. This will include the originals
        #    (due to `results = [image]` in `apply_augmentations`)
        train_images, train_labels = process_letter_batch(
            letter, train_base_images, label, pipeline, augmentations_per_image
        )

        # 2. The test set is *only* the clean, un-augmented base images
        #    Do NOT call process_letter_batch on them.
        test_images = list(test_base_images) # Convert array of images to a list
        test_labels = [label] * len(test_base_images)

        # [--- END FIX ---]

        all_train_images.extend(train_images)
        all_train_labels.extend(train_labels)
        all_test_images.extend(test_images)
        all_test_labels.extend(test_labels)

        # [--- FIX ---] Updated print statement to be accurate
        print(f"   Generated: {len(train_images)} train (augmented), {len(test_images)} test (clean) images")

        # Save sample images for visualization if requested
        if save_samples and len(train_images) >= 10:
            samples_dir = target_dir / "letter_samples"
            samples_dir.mkdir(exist_ok=True)

            # Use train images for samples (since they're more numerous and show augmentations)
            sample_images = train_images[:min(10, len(train_images))]

            # Save first 10 processed images for this letter (original + 9 augmented)
            for i in range(len(sample_images)):
                sample_filename = f"{letter}_{i:02d}.png"
                cv2.imwrite(str(samples_dir / sample_filename), sample_images[i])

            # Also save a comparison grid for this letter
            if len(sample_images) >= 10:
                # Create a 2x5 grid showing original + 9 augmentations
                grid_rows, grid_cols = 2, 5
                # [--- FIX ---] Ensure grid is created with correct dimensions
                grid_img = np.zeros((28 * grid_rows, 28 * grid_cols), dtype=np.uint8)

                for idx in range(min(10, len(sample_images))):
                    row = idx // grid_cols
                    col = idx % grid_cols
                    y_start = row * 28
                    y_end = (row + 1) * 28
                    x_start = col * 28
                    x_end = (col + 1) * 28

                    # [--- FIX ---] Handle potential resize issue if sample_images[idx] is not 28x28
                    # Although apply_augmentations should handle this, it's safer.
                    img_to_place = cv2.resize(sample_images[idx], (28, 28))
                    grid_img[y_start:y_end, x_start:x_end] = img_to_place

                grid_filename = f"{letter}_grid.png"
                cv2.imwrite(str(samples_dir / grid_filename), grid_img)

        # [--- FIX ---] Save a few clean test images for comparison
        if save_samples and len(test_images) > 0:
            samples_dir = target_dir / "letter_samples"
            samples_dir.mkdir(exist_ok=True)
            for i in range(min(5, len(test_images))):
                sample_filename = f"{letter}_CLEAN_TEST_{i:02d}.png"
                cv2.imwrite(str(samples_dir / sample_filename), test_images[i])

    # [--- FIX ---] Shuffle the final datasets to mix letters
    train_combined = list(zip(all_train_images, all_train_labels))
    random.shuffle(train_combined)
    all_train_images, all_train_labels = zip(*train_combined)

    test_combined = list(zip(all_test_images, all_test_labels))
    random.shuffle(test_combined)
    all_test_images, all_test_labels = zip(*test_combined)
    # [--- END FIX ---]

    # Convert to numpy arrays
    train_images = np.array(all_train_images)
    train_labels = np.array(all_train_labels)
    test_images = np.array(all_test_images)
    test_labels = np.array(all_test_labels)

    print(f"\nFinal dataset:")
    print(f"   Train: {len(train_images)} images")
    print(f"   Test: {len(test_images)} images (Clean)")
    print(f"   Total: {len(train_images) + len(test_images)} images")

    # Save in IDX format
    print("\nSaving in IDX format...")

    save_idx_images(target_dir / "font_letters_train-images.idx", train_images)
    save_idx_labels(target_dir / "font_letters_train-labels.idx", train_labels)
    save_idx_images(target_dir / "font_letters_test-images.idx", test_images)
    save_idx_labels(target_dir / "font_letters_test-labels.idx", test_labels)

    # Also save numpy versions for convenience
    print("\nSaving in NPY format...")
    np.save(target_dir / "font_letters_train_images.npy", train_images.astype(np.float32) / 255.0)
    np.save(target_dir / "font_letters_train_labels.npy", train_labels)
    np.save(target_dir / "font_letters_test_images.npy", test_images.astype(np.float32) / 255.0)
    np.save(target_dir / "font_letters_test_labels.npy", test_labels)

    print("\nDataset saved successfully!")
    print(f"Location: {target_dir}")
    print("\nFiles created:")
    print("- font_letters_train-images.idx (IDX format training images)")
    print("- font_letters_train-labels.idx (IDX format training labels)")
    print("- font_letters_test-images.idx (IDX format test images)")
    print("- font_letters_test-labels.idx (IDX format test labels)")
    print("- ...and .npy equivalents")

    # Print statistics (group by unique labels since upper/lower case share labels)
    print("\nLabel distribution:")
    unique_labels = sorted(set(letter_to_label.values()))
    for label in unique_labels:
        # Find all letters that map to this label
        letters_for_label = [letter for letter, lbl in letter_to_label.items() if lbl == label]
        train_count = np.sum(train_labels == label)
        test_count = np.sum(test_labels == label)
        letters_str = '/'.join(letters_for_label)
        print(f"   {letters_str} (label {label}): {train_count} train, {test_count} test")

    # Create a comprehensive overview visualization if samples were saved
    if save_samples:
        print("\nCreating overview visualization...")
        # [--- FIX ---] Pass only the first 26 letters (e.g., lowercase) for a cleaner grid
        create_overview_visualization(target_dir, string.ascii_lowercase)

def create_overview_visualization(target_dir, letters):
    """
    Create a comprehensive overview showing different letters and their augmentations.
    """
    samples_dir = Path(target_dir) / "letter_samples"

    if not samples_dir.exists():
        return

    # [--- FIX ---] Create a grid that fits all 26 letters (e.g., 5x6)
    grid_cols = 6
    grid_rows = int(np.ceil(len(letters) / grid_cols))

    # Create a grid to show one *augmented* sample for each letter
    overview_img_aug = np.full((28 * grid_rows, 28 * grid_cols), 255, dtype=np.uint8)
    # Create a grid to show one *clean test* sample for each letter
    overview_img_clean = np.full((28 * grid_rows, 28 * grid_cols), 255, dtype=np.uint8)

    print("   Generating overview grids (augmented train vs. clean test)...")

    for i, letter in enumerate(letters):
        row = i // grid_cols
        col = i % grid_cols
        y_start = row * 28
        y_end = (row + 1) * 28
        x_start = col * 28
        x_end = (col + 1) * 28

        # Find an augmented sample
        aug_sample_path = samples_dir / f"{letter}_01.png" # 01 is an augmented sample
        if aug_sample_path.exists():
            img = cv2.imread(str(aug_sample_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                overview_img_aug[y_start:y_end, x_start:x_end] = cv2.resize(img, (28, 28))

        # Find a clean test sample
        clean_sample_path = samples_dir / f"{letter}_CLEAN_TEST_00.png"
        if clean_sample_path.exists():
            img = cv2.imread(str(clean_sample_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                overview_img_clean[y_start:y_end, x_start:x_end] = cv2.resize(img, (28, 28))

    # Save the overviews
    cv2.imwrite(str(samples_dir / "overview_AUGMENTED_TRAIN.png"), overview_img_aug)
    cv2.imwrite(str(samples_dir / "overview_CLEAN_TEST.png"), overview_img_clean)
    print(f"   Overviews saved to {samples_dir}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Create large enhanced dataset with sophisticated augmentations
    target_dir = "data/font_letter_dataset_enhanced"
    create_font_based_dataset(target_dir, total_images=100000, train_split=0.85, save_samples=True)