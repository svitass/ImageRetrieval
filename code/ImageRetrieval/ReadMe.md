# File function description #
1. 'MyDataset.py' is used to load data. We use it to load training data set and validation data set.

2. 'predict.py' is used to test the model.

3. 'Siamese3.py' is used to build three streams Siamese architecture.

4. 'trainModel.py' is used to train the network. After all epoch iteration, the model will be saved to'Siamese.pth'.

5. 'resnet50_pre.pth' is the pre-trained model of Resnet50. We use it to initialize the conv of the network

6. 'Siamese.pth' is used to save the trained model, and it can be loaded directly when testing the model.

Note: You can ignore these two files:'model.py' and'train.py'.

# How to run #
1. Training network: 'python trainModel.py'
2. Testing the model: 'python predict.py'