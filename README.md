# Bird Species Embedding and Visualization using TIMM2Spotlight

This project processes a bird species dataset, extracts embeddings using a pretrained model, and visualizes the results using Renumics Spotlight.

## Table of Contents

- [Installation](#installation)
- [Download Dataset](#download-dataset)
- [Usage](#usage)
- [Visualization](#visualization)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/bird-species-visualization.git
   cd bird-species-visualization
   ```

2.	Create and activate a virtual environment. Install all requirements.

## Download Dataset

1.	Download the bird species dataset from Kaggle:

Go to the Kaggle dataset page and download the dataset: [link](https://www.kaggle.com/datasets/gpiosenka/100-bird-species])

2.	Extract the dataset:

Extract the downloaded dataset into the data directory within the project folder. The directory structure should look like this:

```
bird-species-visualization/
├── data/
│   ├── birds.csv
│   ├── test/
│   │   ├── birdClass1/
│   │   │   └── bird1.jpg...
│   │   └── birdClass2/...
│   ├── train/
│   └── valid/
├── dataset.py
├── spotlight_visualization.py
└── README.md
```

## Usage

1.	Run the script to process images and generate embeddings:

```bash
python spotlight_visualization.py
```

This script will:
- Load the bird species dataset.
- Use a pretrained model to generate embeddings for the images.
- Save the results to a CSV file.

2.	Ensure the CSV file is created:

The script will create a file named bird_dataset_predictions.csv in the project directory.

## Visualization

1.	Visualize the results with Renumics Spotlight:

After running the script, the visualization will automatically launch in Renumics Spotlight, displaying the images and their embeddings.

Visualization Example

![<img width="3120" alt="screenshot_visualizatrion" src="https://github.com/user-attachments/assets/d71d60d1-73f7-43e5-9367-915e5038c6d1">](Visualization Example Figure)


