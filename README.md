# CaBMI_analysis

## Dependencies
To use [generalized transfer entropy](https://github.com/olavolav/te-causality), follow the setup instructions in `GTE_Setup.md`. You may also need the newest version of Matplotlib. Other dependencies include pandas, seaborn, scikit-image, scikit-learn, networkx, python-louvain, markov-clustering

## Files
- `pipeline.py`: Wraps the TIFF data (specifically, coming as BIGTIFF data), uses [Caiman](https://github.com/flatironinstitute/CaImAn) to extract ROIs and their activity, and stores the data in HDF5 files.
- `plot_CaBMI`: Scripts to plot ROI activity
- `analysis_CaBMI.py`: Analysis and plotting functions to run on HDF5 files created by the pipeline.
- `utils_cabmi.py`: Utility functions for the analysis scripts.
- `utils_gte.py`: Utility functions for running generalized transfer entropy.
