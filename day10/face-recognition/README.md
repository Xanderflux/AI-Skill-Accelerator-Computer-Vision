# Code for Build Your Own Face Recognition Tool With Python

This is a small project to demonstrate how to build a face recognition tool using Python. It goes along with the Real Python tutorial [Build Your Own Face Recognition Tool With Python](https://realpython.com/face-recognition-with-python/).

## Dependency Installation

`dlib` requires both CMake and a C++ compiler to be installed before using it.
Refer to the [CMake documentation](https://cmake.org/install/) and that of your preferred C++ compiler for
installation instructions.

The script assumes that the directories `training/`, `output/` and `validation/` exist, so be sure to create those directories before running the code.

# Documentation for Face Recognition Script

This Python script implements a face recognition system using the face_recognition library. It supports training a model on known faces, validating the model on known images, and testing it on unknown images. The script uses either the HOG (CPU-based) or CNN (GPU-based) model for face detection and encoding.

## Table of Contents

- [Overview](#overview)
- [Installation Requirements](#installation-requirements)
- [Directory Structure](#directory-structure)
- [How the Script Works](#how-the-script-works)
- [Usage Instructions](#usage-instructions)
  - [Training](#training)
  - [Validation](#validation)
  - [Testing](#testing)
- [Example Commands](#example-commands)
- [Troubleshooting](#troubleshooting)
- [Additional Notes](#additional-notes)

## Overview

The script performs the following tasks:

- **Training**: Encodes faces from images in the training directory and saves the encodings to a pickle file (`output/encodings.pkl`).
- **Validation**: Tests the trained model on images in the validation directory to verify recognition accuracy.
- **Testing**: Recognizes faces in a single provided image and displays the results with bounding boxes and names.

The script uses command-line arguments to control its behavior:

- `--train`: Train the model by encoding faces from the training directory.
- `--validate`: Validate the model using images in the validation directory.
- `--test`: Test the model on a single image specified with the `-f` argument.
- `-m`: Choose the face detection model (`hog` or `cnn`).
- `-f`: Specify the path to an image for testing.

## Installation Requirements

### Prerequisites

- **Python**: Version 3.7 or higher.
- **Operating System**: Windows, macOS, or Linux.
- **Hardware**:
  - For `hog` model: Standard CPU is sufficient.
  - For `cnn` model: A CUDA-capable GPU is recommended for faster processing (requires NVIDIA GPU and CUDA setup).

### Required Python Packages

Install the following packages using pip:

```bash
pip install face_recognition Pillow numpy
```

- **face_recognition**: Provides face detection and encoding functionality. Internally depends on dlib, which may require additional setup (see below).
- **Pillow**: Used for image manipulation and drawing bounding boxes.
- **numpy**: Required for numerical operations in face_recognition.

### Installing dlib (Dependency for face_recognition)

The face_recognition library relies on dlib, which can be tricky to install due to its compilation requirements.

**For CPU-based hog model:**

```bash
pip install dlib
```

**For GPU-based cnn model:**

To use the `cnn` model, you need a CUDA-enabled GPU and dlib compiled with CUDA support. Follow these steps:

1. Install CUDA and cuDNN (if not already installed). Check NVIDIA's website for compatible versions.
2. Install cmake:
   ```bash
   pip install cmake
   ```
3. Install dlib with CUDA support:
   ```bash
   pip install dlib --verbose
   ```
   The `--verbose` flag helps diagnose compilation issues.

**On macOS/Linux:**

You may need to install additional dependencies:

- **Linux:**
  ```bash
  sudo apt-get install libopenblas-dev liblapack-dev libjpeg-dev
  ```
- **macOS:**
  ```bash
  brew install libpng
  ```

**On Windows:**

Install Visual Studio with C++ build tools or use a precompiled dlib wheel if available.

### Optional: Virtual Environment

To avoid dependency conflicts, use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install face_recognition Pillow numpy
```

## Directory Structure

The script expects a specific directory structure for training, validation, and output:

```
project_directory/
├── training/
│   ├── person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── person2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
├── validation/
│   ├── person1/
│   │   ├── image1.jpg
│   ├── person2/
│   │   ├── image1.jpg
├── output/
│   ├── encodings.pkl
├── detector.py
```

- **training/**: Contains subdirectories named after each person (e.g., person1, person2). Each subdirectory holds images of that person.
- **validation/**: Similar structure to training/, used for validating the model.
- **output/**: Stores the `encodings.pkl` file, which contains face encodings and corresponding names.
- **Image Formats**: Supported formats include JPEG, PNG, etc. (compatible with Pillow and face_recognition).

The script automatically creates these directories if they don't exist.

## How the Script Works

### Key Functions

**encode_known_faces(model, encodings_location):**
- Iterates through images in the training directory.
- Detects faces using the specified model (`hog` or `cnn`).
- Generates 128-dimensional face encodings for each detected face.
- Saves encodings and corresponding names to `encodings.pkl`.

**recognize_faces(image_location, model, encodings_location):**
- Loads the input image and detects faces.
- Generates encodings for detected faces.
- Compares these encodings against saved encodings to identify matches.
- Draws bounding boxes and names on the image using Pillow and displays it.

**_recognize_face(unknown_encoding, loaded_encodings):**
- Compares an unknown face encoding against known encodings.
- Uses a voting system (Counter) to determine the most likely match.

**_display_face(draw, bounding_box, name):**
- Draws a blue bounding box around each detected face.
- Adds a white text label with the recognized name (or "Unknown") below the face.

**validate(model):**
- Runs `recognize_faces` on all images in the validation directory to test the model's accuracy.

### Command-Line Arguments

- `--train`: Triggers `encode_known_faces` to train the model.
- `--validate`: Triggers `validate` to test the model on validation images.
- `--test`: Triggers `recognize_faces` on a single image specified with `-f`.
- `-m`: Specifies the model (`hog` or `cnn`). Default: `hog`.
- `-f`: Path to an image for testing.

## Usage Instructions

### Training

1. **Prepare the Training Data:**
   - Place images in the training directory, organized by person (e.g., `training/person1/image1.jpg`).
   - Ensure each image contains a single, clear face of the person.

2. **Run the Training Command:**
   ```bash
   python detector.py --train -m hog
   ```
   - This generates face encodings and saves them to `output/encodings.pkl`.
   - Use `-m cnn` for GPU acceleration if available.

### Validation

1. **Prepare the Validation Data:**
   - Place images in the validation directory, organized similarly to training.
   - These images should contain faces of the same people used in training (for verification).

2. **Run the Validation Command:**
   ```bash
   python detector.py --validate -m hog
   ```
   - The script processes each image in validation, displays the results with bounding boxes and names, and shows whether the model correctly identifies known faces.

### Testing

1. **Prepare a Test Image:**
   - Choose an image with one or more faces (known or unknown).

2. **Run the Testing Command:**
   ```bash
   python detector.py --test -f path/to/test_image.jpg -m hog
   ```
   - The script displays the image with bounding boxes and names (or "Unknown" for unrecognized faces).

### Combining Operations

You can combine flags to perform multiple tasks in one run:

```bash
python detector.py --train --validate -m hog
```

## Example Commands

- **Train the model:**
  ```bash
  python detector.py --train -m hog
  ```

- **Validate the model:**
  ```bash
  python detector.py --validate -m hog
  ```

- **Test on a single image:**
  ```bash
  python detector.py --test -f test.jpg -m hog
  ```

- **Train and validate with CNN model:**
  ```bash
  python detector.py --train --validate -m cnn
  ```

## Troubleshooting

**Module Not Found Error:**
- Ensure all required packages are installed (`face_recognition`, `Pillow`, `numpy`).
- Check that dlib is correctly installed for your platform.

**No Faces Detected:**
- Ensure images are clear, well-lit, and contain visible faces.
- Try switching to the `cnn` model for better accuracy (requires GPU).

**Slow Performance:**
- The `hog` model is slower on large images or many faces. Use the `cnn` model with a GPU for faster processing.
- Resize images to a smaller resolution (e.g., 640x480) before processing.

**CUDA Errors with CNN Model:**
- Verify that CUDA and cuDNN are correctly installed.
- Ensure dlib was compiled with CUDA support.

**Incorrect Recognition:**
- Ensure training images are diverse (different angles, lighting, expressions).
- Add more training images per person for better accuracy.
- Check that validation images match the training data structure.

## Additional Notes

### Model Choice

- **hog**: Faster on CPU, less accurate, no GPU required.
- **cnn**: More accurate, requires GPU and CUDA setup, slower on CPU.

### Data Quality

- Use high-quality images with clear, frontal faces.
- Avoid occlusions (e.g., glasses, hats) unless they are consistent across training and testing.

### Scalability

- For large datasets, consider pre-processing images to a consistent size to reduce memory usage.
- The `encodings.pkl` file can grow large with many faces; ensure sufficient disk space.

### Saving Output

- To save the output image instead of displaying it, modify `recognize_faces` to use `pillow_image.save("output.jpg")` instead of `pillow_image.show()`.

### Extending the Script

- Add support for multiple faces per image in training by modifying `encode_known_faces`.
- Implement confidence scores for matches by using `face_recognition.face_distance`.

### Security

- Be cautious with sensitive data (e.g., personal images). Ensure compliance with privacy regulations.

For further details on face_recognition, refer to its official documentation.