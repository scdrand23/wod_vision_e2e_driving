# WOD 2025 Vision-based End-to-End Driving Challenge

This repository contains code for participating in the [Waymo Open Dataset Vision-based End-to-End Driving Challenge 2025](https://waymo.com/open/challenges/2025/e2e-driving/).

## Project Structure

```
.
├── configs/            # Configuration files (e.g., model hyperparameters, training settings)
├── data/               # Placeholder for downloaded WOD data (ignored by git)
├── notebooks/          # Jupyter notebooks for exploration, visualization, and the original tutorial
│   └── tutorial_vision_based_e2e_driving.py
├── results/            # Output directory for trained models, predictions, logs (ignored by git)
├── scripts/            # Main Python scripts for training, evaluation, submission
│   ├── evaluate.py
│   └── train.py
├── src/                # Source code for models, data loading, utilities
│   └── __init__.py
├── waymo-open-dataset/ # Official WOD repository (submodule or separate clone)
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **(Optional but recommended) Add the official `waymo-open-dataset` repository:**
    You can either clone it separately:
    ```bash
    git clone https://github.com/waymo-research/waymo-open-dataset.git
    ```
    Or add it as a submodule:
    ```bash
    git submodule add https://github.com/waymo-research/waymo-open-dataset.git
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Download the WOD E2E Driving dataset:**
    Visit the [Waymo Open Dataset Website](https://waymo.com/open/) and download the End-to-End Driving Challenge data. Place the downloaded `.tfrecord` files (or extracted data) into the `data/` directory.

## Usage

*   **Training:** Modify and run `scripts/train.py`.
*   **Evaluation:** Modify and run `scripts/evaluate.py`.
*   **Submission:** Adapt the submission generation logic from `notebooks/tutorial_vision_based_e2e_driving.py` into a script.

## Notes

*   Remember to update `requirements.txt` as you add more dependencies.
*   Configure your training runs using files in the `configs/` directory.
*   The `waymo-open-dataset` directory contains useful utilities and proto definitions provided by Waymo.
