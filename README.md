# ML Final Project

`pipline_runner.py` runs the pipeline on all datasets
results is pickle file contains fitted gridsearch, feature scores

`analyze.py` analyze the pickle file

In order to run:
1. Download the dataset and organize it in folders based on the DB name
   for example: bioconductor/ALL.csv
2. install requirement.txt
3. run the pipeline_runner.py
4. in order to run the toy example add one argument `toy` : `pipeline_runner.py toy`