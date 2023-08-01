# Decoupling Recorder Variability from Spatial and Temporal Variation of Ecological Acoustic Indices

## Repository Contents

### Notebooks

Jupyter notebooks for performing different tasks related to the analysis:

- `calcular_coherencia.ipynb`: Calculates coherence between audio signals.
- `calcular_indices.ipynb`: Calculates ecological acoustic indices (EAI) commonly used in soundscape analyses.
- `calcular_metricas.ipynb`: Measures and compares variability among different recording devices and brands.
- `calcular_potencia.ipynb`: Calculates power spectral density (PSD) of audio signals.
- `graficas.ipynb`: Generates graphical representations for visual analysis of the results.
- `proc_indices.ipynb`: Preprocesses and normalizes audio signals for comparison.

### Results

- `figures`: Directory to store figures and results generated during the analysis.

### Utils

Python modules for different utility functions used in the analysis:

- `Filtros.py`: Functions for calculating filters.
- `graphs.py`: Functions for generating graphical representations.
- `indices.py`: Functions for calculating ecological acoustic indices (EAI).
- `sono.py`: Functions for processing audio signals.
- `utils.py`: Other utility functions used in the analysis.

### License

The license file for this repository.

## Functionality

The code in the Jupyter notebooks allows researchers to perform the following tasks:

1. Calculate ecological acoustic indices (EAI) commonly used in soundscape analyses.
2. Measure and compare variability among different recording devices and brands.
3. Calculate power spectral density (PSD) of audio signals.
4. Generate graphical representations for visual analysis of the results.
5. Preprocess and normalize audio signals for comparison.

The proposed preprocessing pipeline reduces EAI variability resulting from different hardware without altering the target information in the audio. It involves three steps:

1. Resample the audio recordings to a common sampling frequency if they have different ones.
2. Determine the comparable frequency range where different devices have similar frequency behavior using coherence.
3. Normalize the signal amplitude to ensure all recordings are in the same range of scales.

For more information and details on how to use the code, refer to the Jupyter notebooks in the `notebooks` directory.

## Authors:
- David Luna-Naranjo
- Juan D. Martínez
- Camilo Sánchez-Giraldo
- Juan M. Daza
- José D. López (Corresponding Author)

This repository contains the code and data used in the analysis presented in the article titled "Decoupling Recorder Variability from Spatial and Temporal Variation of Ecological Acoustic Indices."

For any questions or inquiries regarding the article or the code, feel free to open an issue in this repository, send an email to davidluna.fn@gmail.com, or contact the corresponding author, José D. López, at josedavid@udea.edu.co.
