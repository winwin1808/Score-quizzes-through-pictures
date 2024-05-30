# Student Data Analysis Project

This GitHub repository contains the necessary files for analyzing student data and determining their final results based on performance. The project transforms initial CSV data, processes image data, and utilizes Python scripts and Jupyter notebooks to present outcomes.

<p align="center">
  <img src="DATA/20012322_DaoTien_3A.png" alt="20012322_DaoTien_3A">
</p>

## Project Structure

```
Project_Root/
│
├── DATA/
│   └── (Images and data files)
│
├── ANSWER/
│   └── (1 image related to answers)
│
├── CODE.py
├── CODE.ipynb
├── Final Result.csv
├── README.md
├── Test.py
├── grading.csv
└── student_INFO.csv
```

## Data Files

- **`student_INFO.csv`**
  - Detailed information derived from initial student data.
- **`grading.csv`**
  - Contains grading criteria used to evaluate the students.
- **`Final Result.csv`**
  - Final result (Pass/Fail) for each student in the class.

## Image Data

Images related to the students and project results are stored under the `DATA` and `ANSWER` folders.

## Scripts and Notebooks

- **`CODE.py`**
  - Python script for processing and analyzing data.
- **`CODE.ipynb`**
  - Jupyter Notebook for visual display of the processed results.
- **`Test.py`**
  - Additional testing scripts.

## Libraries

This project utilizes the following Python libraries:

```python
import cv2         # For image processing
import pandas as pd  # For handling CSV data files
import glob        # For retrieving file paths
import imutils     # For image manipulation
import numpy as np  # For numerical operations
import os.path     # For path operations in the filesystem
```

## How to Run

1. Install the necessary dependencies using `pip install -r requirements.txt`.
2. Run `CODE.py` to process the data.
3. Open `CODE.ipynb` in Jupyter Notebook or Lab to view the results.
