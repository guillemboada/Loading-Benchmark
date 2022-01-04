# Benchmark of loading dataset approaches in TensorFlow
While it is possible to input your data to fit a model using a custom _generator_, the `tf.data` API offers possibilities which may be more comfortable and efficient, making use of strategies such as `.cache()` and `.prefetch()` (demonstrated in `CacheAndPrefetch.ipynb`). This repository contains the implementation of four different loading dataset approaches:

* **a)** Using `.from_tensor_slices()`
* **b)** Using `.from_generator()`
* **c)** Using `.flow_from_directory()`
* **d)** Using `.tfrecords`

These are benchmarked on a segmentation task in `DatasetLoadingBenchmark.ipynb`, where the best approaches are discussed. 

## References
To execute the benchmarking on your machine, you will need to:
* Create a virtual environment from `requirements.txt`. This is easily managed executing `pip install -r requirements.txt` on the terminal.
* Install the packages required for GPU computing. Note that the code has only been tested with tensorflow 2.4.0, which works with CUDA 11.0 and cuDNN 8.0. For more combinations, check the [link](https://www.tensorflow.org/install/source#gpu).
* Start a W&B container to log memory usage during training. A quickstart tutorial can be found [here](https://docs.wandb.ai/quickstart).

Please write me if you have any question understanding the code (guillemboada@hotmail.com).

## References
* [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) ([Parkhi et al, 2012](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf))
* [_Image Segmentation TensorFlow tutorial_](https://www.tensorflow.org/tutorials/images/segmentation)
* [_TensorFlow Dataset Pipeline for Semantic Segmentation using tf.data API_, by Idiot Developer](https://www.youtube.com/watch?v=C5CbsTDwQM0)
* [_A hands-on guide to TFRecords_, by Pascal Janetzky](https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c)
* [_Optimize Tensorflow Pipeline Performance: prefetch & cache_, by codebasics](https://www.youtube.com/watch?v=MLEKEplgCas)
