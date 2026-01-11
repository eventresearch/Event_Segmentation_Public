#!/usr/bin/env python3
"""
images_to_video.py

Simple utility to convert a folder of images into a video file using OpenCV.

Features:
- Natural sorting of filenames (so frame_1, frame_2, ..., frame_10 sort correctly)
- Supports PNG/JPG and other image formats by glob
- Configurable FPS, codec, output filename
- Optional resize/width/height
- Progress bar

Usage example:
python3 images_to_video.py /path/to/images output.mp4 --fps 25 --ext png

"""
import argparse
import re
from pathlib import Path
import cv2
from tqdm import tqdm
import sys
import subprocess


def natural_key(s):
    """Key for natural sorting (numbers in filenames sorted numerically)."""
    # split into list of strings and ints
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def find_images(input_dir, ext=None):
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory doesn't exist: {input_dir}")

    # If ext provided, allow comma-separated list
    exts = None
    if ext:
        exts = [e.strip().lower().lstrip('.') for e in ext.split(',') if e.strip()]

    imgs = []
    for f in p.iterdir():
        if not f.is_file():
            continue
        if exts:
            if f.suffix.lower().lstrip('.') in exts:
                imgs.append(f)
        else:
            # accept common image extensions
            if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}:
                imgs.append(f)

    imgs.sort(key=natural_key)
    return imgs


def guess_fourcc(output_path, codec_arg=None):
    if codec_arg:
        return cv2.VideoWriter_fourcc(*codec_arg)
    suf = Path(output_path).suffix.lower()
    if suf in ['.mp4', '.m4v']:
        return cv2.VideoWriter_fourcc(*'mp4v')
    if suf in ['.avi']:
        return cv2.VideoWriter_fourcc(*'XVID')
    # fallback
    return cv2.VideoWriter_fourcc(*'mp4v')


def create_video(image_files, output_path, fps=25, codec=None, width=None, height=None, resize_keep_aspect=False):
    if len(image_files) == 0:
        raise ValueError('No images found to write video.')

    # Read first frame to get size
    first = cv2.imread(str(image_files[0]))
    if first is None:
        raise ValueError(f"Couldn't read first image: {image_files[0]}")

    orig_h, orig_w = first.shape[:2]
    out_w, out_h = orig_w, orig_h
    if width and height:
        out_w, out_h = int(width), int(height)
    elif width and not height:
        out_w = int(width)
        if resize_keep_aspect:
            out_h = int(round(orig_h * (out_w / orig_w)))
    elif height and not width:
        out_h = int(height)
        if resize_keep_aspect:
            out_w = int(round(orig_w * (out_h / orig_h)))

    fourcc = guess_fourcc(output_path, codec)

    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (out_w, out_h))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError('OpenCV VideoWriter failed to open. Try a different codec or output file extension.')

    for p in tqdm(image_files, desc='Writing frames'):
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: skipping unreadable image: {p}")
            continue

        if (img.shape[1], img.shape[0]) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(img)

    writer.release()


def ffmpeg_fallback(input_dir, output_path, fps, ext):
    # ffmpeg requires a sequence pattern; we'll try to use numeric sequence if possible
    # This fallback is best-effort. Inform the user if ffmpeg is not installed.
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps), '-pattern_type', 'glob',
        '-i', f"{input_dir}/*.{ext}", '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError('ffmpeg not found on PATH. Install ffmpeg or choose a different codec/extension.')


def main():
    parser = argparse.ArgumentParser(description='Convert image folder to video')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('output', help='Output video path (e.g., out.mp4)')
    parser.add_argument('--fps', type=float, default=25.0, help='Frames per second')
    parser.add_argument('--ext', type=str, default=None, help='Image extension(s) to include (csv), e.g. png,jpg')
    parser.add_argument('--codec', type=str, default=None, help="FourCC codec (4 chars) like 'mp4v' or 'XVID'")
    parser.add_argument('--width', type=int, default=None, help='Output width in pixels')
    parser.add_argument('--height', type=int, default=None, help='Output height in pixels')
    parser.add_argument('--keep-aspect', action='store_true', help='When resizing keep aspect ratio')
    parser.add_argument('--ffmpeg-fallback', action='store_true', help='If VideoWriter fails, attempt ffmpeg')

    args = parser.parse_args()

    image_files = find_images(args.input_dir, args.ext)
    if len(image_files) == 0:
        print('No images found in', args.input_dir)
        sys.exit(1)

    try:
        create_video(image_files, args.output, fps=args.fps, codec=args.codec, width=args.width, height=args.height, resize_keep_aspect=args.keep_aspect)
        print('Written video to', args.output)
    except RuntimeError as e:
        print('Error:', e)
        if args.ffmpeg_fallback:
            print('Attempting ffmpeg fallback...')
            # choose first extension if provided, else use png
            exts = args.ext.split(',') if args.ext else ['png']
            try:
                ffmpeg_fallback(args.input_dir, args.output, args.fps, exts[0].lstrip('.'))
                print('ffmpeg created video at', args.output)
            except Exception as e2:
                print('ffmpeg fallback failed:', e2)
                sys.exit(2)
        else:
            sys.exit(2)


if __name__ == '__main__':
    main()
