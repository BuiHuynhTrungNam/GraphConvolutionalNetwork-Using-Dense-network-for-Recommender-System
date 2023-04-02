# GraphConvolutionalNetwork-Using-Dense-network-for-Recommender-System
* Source code for using GCN with dense embedding for improve performance of NGCF
* Link NGCF github: https://github.com/xiangwang1223/neural_graph_collaborative_filtering

# Install
* Source code has already changed to new version of python
* If there are any packages that you need to install, use command `pip install <package_name>`

# Structure
* DenseGCF.py is a final file which contains all parts for running code.
* DenseGCF folder contains parts separated from DenseGCF.py file:
	* DenseGCF/DenseGCF.py : main class for denseGCF
	* DenseGCF/utility/batch_test: use for test
	* DenseGCF/utility/helper: use for processing files
	* DenseGCF/utility/load_data: load dataset
	* DenseGCF/utilitymetrics: metrics for model

# Run code
* Place DenseGCF.py and dataset at root, use command "python DenseGCF.py" for running code
* Change name of dataset in line 516 (dataset = "datasetname")
* Result will be write to DenseGCF.txt file


