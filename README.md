# BlurMP

A Python tool to automatically find and mask specified text within video files,
optimized for speed using multiprocessing and Tesseract OCR.

## Features

*   Detects text in video frames using Tesseract OCR (via `pytesseract`).
*   Masks bounding boxes around specified target text strings using:
    *   Gaussian Blur (`--blur-type gaussian`)
    *   Pixelation (`--blur-type pixelate`)
    *   Solid Color Fill (`--blur-type fill`, default black)
*   Uses segment-based processing with object tracking (OpenCV CSRT/KCF) for smoother masking.
*   Utilizes multiprocessing (`-j` option) to process video chunks in parallel across CPU cores, significantly speeding up processing.
*   Requires `ffmpeg` for final video concatenation.
*   Supports time range selection (`--time-range`).
*   Offers retroactive masking (`--backtrack`) to cover initial detection delay.
*   Allows bounding box adjustments (`--x-offset`, `--y-offset`, `--w-scale`, `--h-scale`).
*   Configurable logging levels.
*   Progress indication with ETA for each parallel job.

## Requirements

*   Python 3.8+
*   [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract#installing-tesseract). **Must be installed separately** (e.g., via `brew install tesseract` on macOS, or download from official installers for Windows/Linux).
*   `ffmpeg`. **Must be installed separately** (e.g., via `brew install ffmpeg` on macOS, or download from official website).
*   Python libraries (installed via `pip` or `uv`):
    *   `opencv-python`
    *   `pytesseract`
    *   `numpy`
    *   `loguru`
    *   `rich`

## Installation

It is highly recommended to use a virtual environment.

1.  **Install System Dependencies:**
    *   **Tesseract:** Follow instructions on the [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract#installing-tesseract).
    *   **ffmpeg:** Download from the [ffmpeg website](https://ffmpeg.org/download.html) or use a package manager (`brew install ffmpeg`, `apt install ffmpeg`, etc.).

2.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3.  **Create and activate a virtual environment:**
    *   Using `venv`:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate  # Linux/macOS
        # .\.venv\Scripts\activate # Windows
        ```
    *   Using `uv` (recommended):
        ```bash
        uv venv
        source .venv/bin/activate  # Linux/macOS
        # .\.venv\Scripts\activate # Windows
        ```

4.  **Install Python dependencies:**
    *   Using `pip`:
        ```bash
        pip install -e .
        ```
    *   Using `uv` (recommended):
        ```bash
        uv pip install -e .
        # or: uv pip sync pyproject.toml
        ```

## Usage

```bash
python blurry_mp.py <input_video_path> <output_video_path> [-t TEXT] [-f FILE] [options...]
```

Or, if installed via the `[project.scripts]` entry point:

```bash
blurmp <input_video_path> <output_video_path> [-t TEXT] [-f FILE] [options...]
```

**Required Arguments:**

*   `input_video`: Path to the input video file.
*   `output_video`: Path where the processed video will be saved.
*   At least one text target via `-t`/`--texts` or `-f`/`--strings-file`.

**Key Options:**

*   `-t TEXT`, `--texts TEXT`: Target string(s) to mask (repeat for multiple).
*   `-f FILE`, `--strings-file FILE`: Path to a file with one target string per line.
*   `-sf N`, `--segment-frames N`: Analyze 1 frame every N frames (default: 15). Lower is more reactive but slower.
*   `--backtrack N`: Mask N previous frames retroactively (default: 0). Useful if `N >= segment-frames`.
*   `--blur-type {gaussian,pixelate,fill}`: Type of masking effect (default: gaussian).
*   `-k N`, `--kernel-size N`: Kernel size for Gaussian blur (default: 51).
*   `--pixel-size N`: Block size for pixelation (default: 16). Larger is coarser.
*   `--fill-color B,G,R`: BGR color for fill type (default: "0,0,0" - black).
*   `-j N`, `--jobs N`: Number of parallel processes (default: CPU cores).
*   `--time-range START-END`: Process only specific time ranges (e.g., "0:01:10-0:02:30", "30-90"). Can be repeated.
*   `--draw-boxes`: Draw red boxes instead of masking (for debugging).
*   `--x-offset P`, `--y-offset P`, `--w-scale F`, `--h-scale F`: Adjust bounding box position and size.
*   `--log-level LEVEL`: Set logging verbosity (DEBUG, INFO, etc.).

**Example:**

```bash
# Process using 8 cores, pixelation, targeting text from a file,
# analyzing every 3 frames with 3 backtrack frames.
python blurry_mp.py input.mp4 output_pixel.mp4 \
    -f sensitive_words.txt \
    --segment-frames 3 \
    --backtrack 3 \
    --blur-type pixelate \
    --pixel-size 20 \
    -j 8

# Process using black fill boxes
blurmp input.mov output_fill.mp4 \
    -t "Secret Project" -t "Password123" \
    --blur-type fill \
    -j 16
```

## How It Works

1.  **Chunking:** The video's total frame count is divided among the number of specified parallel jobs (`-j`).
2.  **Parallel Processing:** Each job processes its assigned chunk of frames:
    *   Reads frames sequentially within its chunk.
    *   At the start of each analysis segment (`-sf`), performs Tesseract OCR on the current frame to detect target text.
    *   Initializes OpenCV trackers (e.g., CSRT) for detected regions.
    *   For subsequent frames within the segment, updates trackers to follow the text.
    *   Applies the chosen masking effect (blur, pixelation, or fill) to the regions identified by detection or tracking.
    *   Optionally applies the mask retroactively (`--backtrack`) to frames held in a buffer to compensate for detection delay.
    *   Writes the processed frames to a temporary video file for that chunk.
3.  **Concatenation:** Once all parallel jobs complete, `ffmpeg` is used to concatenate the temporary chunk files into the final output video without re-encoding (`-c copy`).
4.  **Cleanup:** Temporary files and directories are removed.

## Limitations

*   **OCR Accuracy:** Tesseract's accuracy depends on text quality, font, size, and orientation. Pre-processing helps but isn't perfect.
*   **Tracker Drift/Failure:** Object trackers can lose the target, especially with fast motion, rotations, or occlusions. This might cause the mask to disappear briefly between analysis segments.
*   **Performance:** While multiprocessing helps significantly, OCR remains CPU-intensive. Processing very high-resolution video or using very small `segment-frames` values can still be time-consuming.
*   **Dependencies:** Requires external installation of Tesseract and ffmpeg.

## License

MIT License (see LICENSE file). 