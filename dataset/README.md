Place 'data.mat' from [link](https://springernature.figshare.com/collections/A_database_of_human_gait_performance_on_irregular_and_uneven_surfaces_collected_by_wearable_sensors/4892463) of the dataset from [A database of human gait performance on irregular and uneven surfaces collected by wearable sensors](https://www.nature.com/articles/s41597-020-0563-y)

and run in order: 
```
matlab mat_to_py_cmptbl_mat.m
python py-cmptbl_mat_to_npy.py
```
, to generate Gait on Irregular Surface(GoIS) dataset used in the research questions.

Run: 
```
python normalized_dataset.py
```
to generate normalized GoIS(nGoIS)