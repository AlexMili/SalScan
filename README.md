# SalScan

SalScan is a universal framework to develop, compare and visualize Saliency models.

## Install

This module has only been tested under **Python 3.8**.

Use the following command to install dependencies:
```python
pip install -r requirements.txt
```
And install module in edit mode with the following command:
```python
pip install -e .
```


## Usage

### Datasets
Before launching any model, you will need a dataset. A specific class need to be created for each dataset you will use. But before creating your own class, check if your dataset is available in `SalScan.Dataset`. For now, **CAT2000**, **Le Meur** and **Toronto** datasets are supported.

If you want to create your own dataset class here are some points of attention:

- The name of your class needs to have the same name as its containing file and it must end with the `Dataset` keyword: `dataset_nameDataset`
- Your class will take as an input the path of the dataset. This path must point to the original files provided by its author. Thus, anyone with this class will be able to use it in this framework without any further processing.
- Your class must inherit from `AbstractDataset`.
- Your class need to implement all methods as in already existing datasets in order to work.

Go check the code from already existing datasets and `AbstractDataset` to develop your own.

A dataset is initialised with its path:
```python
from SalScan.Dataset.CAT2000Dataset import CAT2000Dataset

cat2000 = CAT2000Dataset(path="/path/to/cat200/dataset")
```

You can then populate your dataset by scanning the directory and load data:
```python
cat2000.populate()
```
Note that this is done automatically in sessions.

### Sessions

A model can be run in standalone but with **SalScan** you can use sessions. A session is a unique association of a dataset, a model, parameters and metrics. You will need the three to run a session. Sessions are generic and does not need to be re-implemented or edited. Heres is how you can use them:

### Models
Here goes how to use models in standalone.

### Metrics
Metrics available.


# Acknowledgment

This work has been started during Alexandre Milisavljevic PhD thesis and extended by [Alessandro Mondin](https://github.com/AlessandroMondin) through the support of [Ittention](https://www.ittention.com).

# Citation
```
@PHDTHESIS{mili2020,
    url = "http://www.theses.fr/2020UNIP5136",
    title = "Visual exploration of web pages : understanding its dynamic for a better modelling",
    author = "Milisavljevic, Alexandre",
    year = "2020",
}
```
