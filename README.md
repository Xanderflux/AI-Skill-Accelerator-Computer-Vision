# AI-Skill-Accelerator-Computer-Vision
![img1](./images/logo.png)

Society for AI Accelerator Program for Computer Vision

# 🛠️ Setup Guide: Python + OpenCV with Anaconda (Windows/macOS/Linux)

This guide helps you install Python, OpenCV, and supporting libraries using Anaconda on any major OS.

---

## ✅ Step 1: Install Anaconda (Python 3.x)

1. Go to the official website: https://www.anaconda.com/products/distribution
2. Download the installer for your OS:
   - **Windows**: `.exe`
   - **macOS**: `.pkg`
   - **Linux**: `.sh`
3. Run the installer and follow the instructions (default options are fine).

---

## ✅ Step 2: Create a New Anaconda Environment (Recommended)

Use a new environment to isolate dependencies.

```bash
conda create -n cv_env python=3.10
```

Replace `cv_env` with your preferred environment name.

Then activate the environment:

# Windows
conda activate cv_env

# macOS/Linux
conda activate cv_env

## ✅ Step 3: Install OpenCV and Supporting Libraries

Install OpenCV via `conda` (from `conda-forge`) or `pip`:

🔹 Option 1: Using conda-forge (recommended)

```bash
conda install -c conda-forge opencv matplotlib numpy
```

🔹 Option 2: Using pip

```bash
pip install opencv-python matplotlib numpy
```

⚠️ If you need GUI functions (like `cv2.imshow`), also install GUI backends:

* **Windows**: Usually works out of the box.
* **macOS**: Install `qt`:

```bash
conda install -c conda-forge pyqt
```

* **Linux**: Make sure `libgtk2.0-dev` or `libgtk-3-dev` is installed via `apt`.

## ✅ Step 4: Test Your Installation

Create a file named `test_cv.py`:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((300, 300, 3), dtype='uint8')
image[:] = (255, 0, 0)  # Blue in BGR

cv2.imshow("Blue Window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Run it:

```bash
python test_cv.py
```

If using Jupyter Notebook:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((300, 300, 3), dtype='uint8')
image[:] = (255, 0, 0)  # Blue in BGR

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

## ✅ Install Jupyter Notebook

```bash
conda install notebook
```

Then run:

```bash
jupyter notebook
```

🧼 Bonus: Deactivate or Remove Environment

To deactivate:

```bash
conda deactivate
```

