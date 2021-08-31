Divide the data into train dataset, val dataset:'python split_data.py'

If you need to test a new data set, make sure that your folder structure is the same as the one given.

Note: We keep part of the data as test data when dividing. Because the structure of the test data is different from the structure of these two data sets, we choose to construct the test dataset manually.

**When constructing the test dataset, note that the non-Relevant image and shop image are both from the shop domain, and the query image is from the user domain.**

Tips:In order to comply with the rules of using the DeepFashion dataset, please do not arbitrarily spread the dataset.