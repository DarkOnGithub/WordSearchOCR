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
    small_noise_map = np.random.rand(low_res_h, low_res_w) * 2 - 1
    noise_map = cv2.resize(small_noise_map, (width, height), interpolation=cv2.INTER_CUBIC)
    return noise_map

class CombinedCrumplePrintTransform(ImageOnlyTransform):

    def __init__(self,
                 crumple_strength=12,
                 crease_intensity=140,
                 crumple_scale_w=24,
                 crumple_scale_h=18,
                 blur_kernel=(3, 3),
                 contrast_alpha=0.8,
                 brightness_beta=30,
                 noise_scale=4,
                 canvas_scale=2.8,
                 ink_bleed_strength=0.25,
                 paper_yellowing=0.12,
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

        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    def apply(self, image, **params):
        height, width = image.shape[:2]

        canvas_height = int(height * self.canvas_scale)
        canvas_width = int(width * self.canvas_scale)

        canvas = np.random.normal(240, 10, (canvas_height, canvas_width)).astype(np.uint8)
        if len(image.shape) == 3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        max_offset_h = canvas_height - height
        max_offset_w = canvas_width - width

        if max_offset_h > 0 and max_offset_w > 0:
            offset_h = np.random.randint(0, max_offset_h)
            offset_w = np.random.randint(0, max_offset_w)
        else:
            offset_h = offset_w = 0

        canvas[offset_h:offset_h+height, offset_w:offset_w+width] = image

        processed = cv2.blur(canvas, self.blur_kernel)
        processed = cv2.convertScaleAbs(processed, alpha=self.contrast_alpha, beta=self.brightness_beta)

        if self.ink_bleed_strength > 0:
            ink_bleed = cv2.dilate(processed, self._morph_kernel, iterations=1)
            processed = cv2.addWeighted(processed, 1-self.ink_bleed_strength, ink_bleed, self.ink_bleed_strength, 0)

        if self.paper_yellowing > 0:
            yellow_overlay = np.full_like(processed, [240, 235, 220], dtype=np.uint8)
            if len(processed.shape) == 3:
                processed = cv2.addWeighted(processed, 1-self.paper_yellowing, yellow_overlay, self.paper_yellowing, 0)
            else:
                yellow_factor = 1 - self.paper_yellowing * 0.1
                processed = cv2.convertScaleAbs(processed, alpha=yellow_factor, beta=self.paper_yellowing * 10)

        noise_x = create_low_freq_noise(canvas_width, canvas_height, self.crumple_scale_w, self.crumple_scale_h)
        noise_y = create_low_freq_noise(canvas_width, canvas_height, self.crumple_scale_w, self.crumple_scale_h)

        zone_map = create_low_freq_noise(canvas_width, canvas_height, 6, 5)
        zone_mask = cv2.normalize(zone_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
        zone_mask = np.power(zone_mask, 2)
        zone_mask = cv2.GaussianBlur(zone_mask, (51, 51), 0)

        displacement_x = noise_x * self.crumple_strength * zone_mask
        displacement_y = noise_y * self.crumple_strength * zone_mask

        x_coords, y_coords = np.meshgrid(np.arange(canvas_width), np.arange(canvas_height))
        map_x = (x_coords + displacement_x).astype(np.float32)
        map_y = (y_coords + displacement_y).astype(np.float32)

        warped_canvas = cv2.remap(processed, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        noise_y_32f = noise_y.astype(np.float32)
        grad_x = cv2.Sobel(noise_y_32f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(noise_y_32f, cv2.CV_32F, 0, 1, ksize=3)
        crease_map = np.abs(grad_x) + np.abs(grad_y)

        shading_map = cv2.normalize(crease_map, None, 0, 1.0, cv2.NORM_MINMAX)
        shading_map = shading_map * self.crease_intensity * zone_mask

        warped_with_shading = warped_canvas.astype(np.int32) + shading_map.astype(np.int32)
        warped_with_shading = np.clip(warped_with_shading, 0, 255).astype(np.uint8)

        grain_noise = np.random.normal(loc=0, scale=self.noise_scale, size=warped_with_shading.shape).astype(np.int16)
        final_canvas = np.clip(warped_with_shading.astype(np.int16) + grain_noise, 0, 255).astype(np.uint8)

        result = final_canvas[offset_h:offset_h+height, offset_w:offset_w+width]

        return result

def get_common_fonts():
    FONT_WHITELIST = [
        "times",
        "georgia",
        "cambria",
        "NotoSerif",
        "Merriweather",
        "Garamond",
        "DroidSerif",
        "PTSerif",
        "Alegreya",
        "Vollkorn",
        "Crimson",
        "Cormorant",
        "Baskerville",
        "Minion",
        "Caslon",
        "LiberationSerif",
        "DejaVuSerif",
        "Libertine",

        "cour",
        "consola",
        "lucon",
        "Inconsolata",
        "JetBrainsMono",
        "FiraCode",
        "SourceCodePro",
        "NotoMono",
        "UbuntuMono",
        "DroidSansMono",
        "RobotoMono",
        "IBMPlexMono",
        "LiberationMono",
        "DejaVuSansMono",

        "FiraSans",
        "PTSans",
        "Inter",
        "Roboto",
        "OpenSans",
        "Lato",
        "IBMPlexSans",
        "arial.ttf", "Arial.ttf", "ARIAL.TTF",
        "arialbd.ttf", "Arialbd.ttf", "ARIALBD.TTF",
        "arialbi.ttf", "Arialbi.ttf", "ARIALBI.TTF",
        "ariali.ttf", "Ariali.ttf", "ARIALI.TTF",

        "times.ttf", "Times.ttf", "TIMES.TTF",
        "timesbd.ttf", "Timesbd.ttf", "TIMESBD.TTF",
        "timesbi.ttf", "Timesbi.ttf", "TIMESBI.TTF",
        "timesi.ttf", "Timesi.ttf", "TIMESI.TTF",

        "cour.ttf", "Cour.ttf", "COUR.TTF",
        "courbd.ttf", "Courbd.ttf", "COURBD.TTF",
        "courbi.ttf", "Courbi.ttf", "COURBI.TTF",
        "couri.ttf", "Couri.ttf", "COURI.TTF",

        "calibri.ttf", "Calibri.ttf", "CALIBRI.TTF",
        "calibrib.ttf", "Calibrib.ttf", "CALIBRIB.TTF",
        "calibrii.ttf", "Calibrii.ttf", "CALIBRII.TTF",
        "calibril.ttf", "Calibril.ttf", "CALIBRIL.TTF",

        "tahoma.ttf", "Tahoma.ttf", "TAHOMA.TTF",
        "tahomabd.ttf", "Tahomabd.ttf", "TAHOMABD.TTF",

        "verdana.ttf", "Verdana.ttf", "VERDANA.TTF",
        "verdanab.ttf", "Verdanab.ttf", "VERDANAB.TTF",
        "verdanai.ttf", "Verdanai.ttf", "VERDANAI.TTF",
        "verdanaz.ttf", "Verdanaz.ttf", "VERDANAZ.TTF",

        "georgia.ttf", "Georgia.ttf", "GEORGIA.TTF",
        "georgiab.ttf", "Georgiab.ttf", "GEORGIAB.TTF",
        "georgiai.ttf", "Georgiai.ttf", "GEORGIAI.TTF",
        "georgiaz.ttf", "Georgiaz.ttf", "GEORGIAZ.TTF",

        "Helvetica.ttf", "helvetica.ttf", "HELVETICA.TTF",
        "Helvetica-Bold.ttf", "helvetica-bold.ttf",

        "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
        "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
        "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf",

        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
        "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
        "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf",

        "Ubuntu-Regular.ttf", "Ubuntu-Bold.ttf",
        "UbuntuMono-Regular.ttf", "UbuntuMono-Bold.ttf",

        "segoeui.ttf", "Segoeui.ttf", "SEGOEUI.TTF",
        "segoeuib.ttf", "Segoeuib.ttf", "SEGOEUIB.TTF",
        "segoeuil.ttf", "Segoeuil.ttf", "SEGOEUIL.TTF",

        "cambria.ttc", "Cambria.ttc", "CAMBRIATTC",
        "cambriab.ttf", "Cambriab.ttf", "CAMBRIAB.TTF",

        "consola.ttf", "Consola.ttf", "CONSOLA.TTF",
        "consolab.ttf", "Consolab.ttf", "CONSOLAB.TTF",

        "lucon.ttf", "Lucon.ttf", "LUCON.TTF",

        "system.ttf", "System.ttf", "SYSTEM.TTF",
        "sfns.ttf", "Sfns.ttf", "SFNS.TTF",
    ]

    font_dirs = [
        "C:/Windows/Fonts/",
        "C:/WINNT/Fonts/",

        "/System/Library/Fonts/",
        "/Library/Fonts/",
        "/System/Library/Assets/com_apple_MobileAsset_Font5/",
        "~/Library/Fonts/",

        "/usr/share/fonts/",
        "/usr/share/fonts/truetype/",
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
        "/usr/share/fonts/truetype/ubuntu/",
        "/usr/share/fonts/truetype/freefont/",
        "/usr/share/fonts/TTF/",
        "/usr/share/fonts/OTF/",
        "/usr/local/share/fonts/",

        "/usr/X11R6/lib/X11/fonts/TTF/",
        "/usr/X11R6/lib/X11/fonts/truetype/",
        "/var/lib/defoma/x-ttcidfont-conf.d/dirs/TrueType/",

        "/usr/local/share/fonts/",

        "/system/fonts/",

        "./fonts/",
        "../fonts/",
        "../../fonts/",
    ]

    available_fonts = []

    for font_dir in font_dirs:
        try:
            expanded_dir = os.path.expanduser(font_dir)
            if os.path.exists(expanded_dir):
                for font_file in FONT_WHITELIST:
                    font_path = os.path.join(expanded_dir, font_file)
                    if os.path.exists(font_path):
                        available_fonts.append(font_path)
        except Exception as e:
            print(f"Warning: Could not search directory {font_dir}. Error: {e}")

    seen = set()
    available_fonts = [x for x in available_fonts if not (x in seen or seen.add(x))]

    print(f"Found {len(available_fonts)} system fonts")

    if not available_fonts:
        try:
            ImageFont.load_default()
            available_fonts = ["default"]
            print("Using PIL default font")
        except Exception as e:
            print(f"Warning: PIL default font failed: {e}")

    if not available_fonts:
        print("Warning: No fonts found. Creating synthetic fonts.")
        available_fonts = ["synthetic"]

    return available_fonts

def generate_synthetic_font(letter, font_size=24, image_size=28):
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (image_size - text_width) // 2 - bbox[0]
        y = (image_size - text_height) // 2 - bbox[1]

        draw.text((x, y), letter, fill=0, font=font)
    except:
        if letter.isupper():
            draw.rectangle([8, 6, 20, 22], fill=0)
        else:
            draw.rectangle([8, 10, 20, 22], fill=0)

    return np.array(img).astype(np.uint8)

def generate_letter_image(letter, font_path, font_size=24, image_size=28):
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    try:
        if font_path == "default":
            font = ImageFont.load_default()
        elif font_path == "synthetic":
            return generate_synthetic_font(letter, font_size, image_size)
        else:
            font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        try:
            font = ImageFont.load_default()
        except:
            return generate_synthetic_font(letter, font_size, image_size)

    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (image_size - text_width) // 2 - bbox[0]
    y = (image_size - text_height) // 2 - bbox[1]

    draw.text((x, y), letter, fill=0, font=font)

    img_array = np.array(img)

    return img_array.astype(np.uint8)

def create_base_letter_images(letters, fonts, images_per_letter=50):
    base_images = {}

    for letter in letters:
        print(f"Generating base images for letter '{letter}'...")
        letter_images = []

        for _ in range(images_per_letter):
            font_path = random.choice(fonts)

            font_size = random.randint(18, 32)

            img = generate_letter_image(letter, font_path, font_size)

            letter_images.append(img)

        base_images[letter] = np.array(letter_images)
        print(f"   Generated {len(letter_images)} base images for '{letter}'")

    return base_images

def create_augmentation_pipeline():
    return Compose([
        CombinedCrumplePrintTransform(
            crumple_strength=12, crease_intensity=140,
            crumple_scale_w=24, crumple_scale_h=18,
            blur_kernel=(3, 3), contrast_alpha=0.8, brightness_beta=30,
            noise_scale=4, canvas_scale=2.8,
            ink_bleed_strength=0.25, paper_yellowing=0.12,
            p=0.85
        ),

        A.Rotate(limit=17, border_mode=cv2.BORDER_CONSTANT, fill=255, p=0.75),
        A.Perspective(scale=(0.045, 0.11), p=0.65),
        A.Affine(scale=(0.78, 1.22), translate_percent=(-0.12, 0.12),
                rotate=(-6, 6), shear=(-6, 6), p=0.7),

        A.GaussNoise(std_range=(0.02, 0.12), p=0.4),
        A.GaussianBlur(blur_limit=(3, 3), p=0.15),
        A.MotionBlur(blur_limit=(3, 3), p=0.12),
        A.Defocus(radius=(1, 1), alias_blur=(0.05, 0.15), p=0.08),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.35),

        A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15),
                                  contrast_limit=(-0.15, 0.15), p=0.45),

        A.GridDistortion(num_steps=3, distort_limit=0.12, p=0.2),
        A.ElasticTransform(alpha=2.0, sigma=50, p=0.18),
        A.OpticalDistortion(distort_limit=0.08, p=0.14),

        A.GaussNoise(std_range=(0.005, 0.07), p=0.35),
        A.Sharpen(alpha=(0.0, 0.25), lightness=(0.8, 1.2), p=0.18),
        A.UnsharpMask(alpha=(0.0, 0.18), threshold=10, p=0.14),

        A.GridDropout(ratio=0.3, p=0.08),
        A.CoarseDropout(p=0.08),
    ])

def apply_augmentations(image, pipeline, num_augmentations=6):
    results = [image]

    results.extend([None] * num_augmentations)

    for i in range(num_augmentations):
        augmented = pipeline(image=image)
        aug_img = augmented['image']
        if aug_img.shape[:2] != (28, 28):
            aug_img = cv2.resize(aug_img, (28, 28), interpolation=cv2.INTER_LINEAR)
        results[i + 1] = aug_img.astype(np.uint8)

    return results

def process_letter_batch(letter, images, label, pipeline, augmentations_per_image):
    total_images = len(images) * (augmentations_per_image + 1)

    all_images = []
    all_labels = [label] * total_images

    for img in images:
        augmented_images = apply_augmentations(img, pipeline, augmentations_per_image)
        all_images.extend(augmented_images)

    return all_images, all_labels

def process_single_letter(args):
    letter, base_images, label, pipeline, augmentations_per_image, train_split, save_samples = args

    indices = list(range(len(base_images)))
    random.shuffle(indices)
    base_images = base_images[indices]

    split_idx = int(len(base_images) * train_split)
    train_base_images = base_images[:split_idx]
    test_base_images = base_images[split_idx:]

    train_images, train_labels = process_letter_batch(
        letter, train_base_images, label, pipeline, augmentations_per_image
    )

    test_images, test_labels = process_letter_batch(
        letter, test_base_images, label, pipeline, augmentations_per_image
    )

    sample_data = None
    if save_samples and len(train_images) >= 10 and letter.islower():
        sample_images = train_images[:min(10, len(train_images))]
        sample_data = {
            'letter': letter,
            'samples': sample_images,
            'test_samples': test_images[:min(5, len(test_images))] if test_images else []
        }

    return letter, train_images, train_labels, test_images, test_labels, sample_data

def save_idx_images(filename, images):
    num_images, height, width = images.shape
    magic = 2051
    header = struct.pack('>IIII', magic, num_images, height, width)

    with open(filename, 'wb') as f:
        f.write(header)
        for img in images:
            f.write(img.tobytes())

def save_idx_labels(filename, labels):
    num_labels = len(labels)
    magic = 2049
    header = struct.pack('>II', magic, num_labels)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(labels.astype(np.uint8).tobytes())

def create_font_based_dataset(target_dir, total_images=100000, train_split=0.85, save_samples=False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    letters = string.ascii_letters
    letter_to_label = {letter: ord(letter.lower()) - ord('a') for letter in letters}

    unique_classes = len(set(letter_to_label.values()))
    print(f"Creating dataset with {unique_classes} letter classes (upper/lower case share labels)")
    print(f"Processing {len(letters)} letter variants: {letters}")
    print(f"Target: {total_images} total images ({train_split*100:.0f}% train, {(1-train_split)*100:.0f}% test)")

    fonts = get_common_fonts()
    if not fonts or fonts == ["synthetic"]:
         print("CRITICAL WARNING: No real fonts found. Dataset will be low quality.")
    print(f"Using {len(fonts)} available fonts")

    images_per_letter = total_images // len(letters)
    base_images_per_letter = max(60, images_per_letter // 32)

    print(f"Generating {base_images_per_letter} base images per letter variant, then augmenting to ~{images_per_letter} total per variant")

    print("\nGenerating base letter images...")
    base_images = create_base_letter_images(letters, fonts, base_images_per_letter)

    pipeline = create_augmentation_pipeline()

    augmentations_per_image = max(0, (images_per_letter // (base_images_per_letter + 1)) - 1)

    print(f"\nSplitting base images and augmenting ({augmentations_per_image} augmentations per base train image)...")

    process_args = []
    for letter in letters:
        base_letter_images = base_images[letter]
        label = letter_to_label[letter]
        process_args.append((
            letter, base_letter_images, label, pipeline,
            augmentations_per_image, train_split, save_samples
        ))

    print(f"\nProcessing {len(letters)} letters using {min(mp.cpu_count(), len(letters))} parallel workers...")

    all_train_images = []
    all_train_labels = []
    all_test_images = []
    all_test_labels = []
    all_sample_data = []

    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(letters))) as executor:
        futures = [executor.submit(process_single_letter, args) for args in process_args]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing letters"):
            try:
                letter, train_images, train_labels, test_images, test_labels, sample_data = future.result()

                print(f"   Letter '{letter}' (label {letter_to_label[letter]}): {len(train_images)} train, {len(test_images)} test")

                all_train_images.extend(train_images)
                all_train_labels.extend(train_labels)
                all_test_images.extend(test_images)
                all_test_labels.extend(test_labels)

                if sample_data:
                    all_sample_data.append(sample_data)

            except Exception as e:
                print(f"Error processing letter: {e}")
                raise

    if save_samples and all_sample_data:
        print("\nSaving sample images...")
        samples_dir = target_dir / "letter_samples"
        samples_dir.mkdir(exist_ok=True)

        for sample_data in tqdm(all_sample_data, desc="Saving samples"):
            letter = sample_data['letter']
            sample_images = sample_data['samples']
            test_samples = sample_data['test_samples']

            for i in range(len(sample_images)):
                sample_filename = f"{letter}_{i:02d}.png"
                cv2.imwrite(str(samples_dir / sample_filename), sample_images[i])

            if len(sample_images) >= 10:
                grid_rows, grid_cols = 2, 5
                grid_img = np.zeros((28 * grid_rows, 28 * grid_cols), dtype=np.uint8)

                for idx in range(min(10, len(sample_images))):
                    row = idx // grid_cols
                    col = idx % grid_cols
                    y_start = row * 28
                    y_end = (row + 1) * 28
                    x_start = col * 28
                    x_end = (col + 1) * 28

                    img_to_place = cv2.resize(sample_images[idx], (28, 28))
                    grid_img[y_start:y_end, x_start:x_end] = img_to_place

                grid_filename = f"{letter}_grid.png"
                cv2.imwrite(str(samples_dir / grid_filename), grid_img)

            for i in range(len(test_samples)):
                sample_filename = f"{letter}_CLEAN_TEST_{i:02d}.png"
                cv2.imwrite(str(samples_dir / sample_filename), test_samples[i])

    train_combined = list(zip(all_train_images, all_train_labels))
    random.shuffle(train_combined)
    all_train_images, all_train_labels = zip(*train_combined)

    test_combined = list(zip(all_test_images, all_test_labels))
    random.shuffle(test_combined)
    all_test_images, all_test_labels = zip(*test_combined)

    train_images = np.array(all_train_images)
    train_labels = np.array(all_train_labels)
    test_images = np.array(all_test_images)
    test_labels = np.array(all_test_labels)

    print(f"\nFinal dataset:")
    print(f"   Train: {len(train_images)} images")
    print(f"   Test: {len(test_images)} images (Clean)")
    print(f"   Total: {len(train_images) + len(test_images)} images")

    print("\nSaving in IDX format...")

    save_idx_images(target_dir / "font_letters_train-images.idx", train_images)
    save_idx_labels(target_dir / "font_letters_train-labels.idx", train_labels)
    save_idx_images(target_dir / "font_letters_test-images.idx", test_images)
    save_idx_labels(target_dir / "font_letters_test-labels.idx", test_labels)

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

    print("\nLabel distribution:")
    unique_labels = sorted(set(letter_to_label.values()))
    for label in unique_labels:
        letters_for_label = [letter for letter, lbl in letter_to_label.items() if lbl == label]
        train_count = np.sum(train_labels == label)
        test_count = np.sum(test_labels == label)
        letters_str = '/'.join(letters_for_label)
        print(f"   {letters_str} (label {label}): {train_count} train, {test_count} test")

    if save_samples:
        print("\nCreating overview visualization...")
        create_overview_visualization(target_dir, string.ascii_lowercase)

def create_overview_visualization(target_dir, letters):
    """
    Create a comprehensive overview showing different letters and their augmentations.
    """
    samples_dir = Path(target_dir) / "letter_samples"

    if not samples_dir.exists():
        return

    grid_cols = 6
    grid_rows = int(np.ceil(len(letters) / grid_cols))

    overview_img_aug = np.full((28 * grid_rows, 28 * grid_cols), 255, dtype=np.uint8)
    overview_img_clean = np.full((28 * grid_rows, 28 * grid_cols), 255, dtype=np.uint8)

    print("   Generating overview grids (augmented train vs. clean test)...")

    for i, letter in enumerate(letters):
        row = i // grid_cols
        col = i % grid_cols
        y_start = row * 28
        y_end = (row + 1) * 28
        x_start = col * 28
        x_end = (col + 1) * 28

        aug_sample_path = samples_dir / f"{letter}_01.png"
        if aug_sample_path.exists():
            img = cv2.imread(str(aug_sample_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                overview_img_aug[y_start:y_end, x_start:x_end] = cv2.resize(img, (28, 28))

        clean_sample_path = samples_dir / f"{letter}_CLEAN_TEST_00.png"
        if clean_sample_path.exists():
            img = cv2.imread(str(clean_sample_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                overview_img_clean[y_start:y_end, x_start:x_end] = cv2.resize(img, (28, 28))

    cv2.imwrite(str(samples_dir / "overview_AUGMENTED_TRAIN.png"), overview_img_aug)
    cv2.imwrite(str(samples_dir / "overview_CLEAN_TEST.png"), overview_img_clean)
    print(f"   Overviews saved to {samples_dir}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    target_dir = "data"
    create_font_based_dataset(target_dir, total_images=200000, train_split=0.85, save_samples=True)