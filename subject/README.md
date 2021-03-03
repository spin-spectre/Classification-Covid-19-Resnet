## DIP Course Project-[Covid,Cap,Normal] Image Classification with CNN

### Preparations

The platform I used for training is tensorflow-gpu 2.0.0-alpha0.Please ensure 
that you've already configured with the specific tf version(otherwise some errors 
may occur).

### Training
The code is annotated with detailed informations.

#### Dataset
Your can import your own dataset by changing the path of the variable 
'data_path'.Please ensure that the structure of your dataset is the same as below:
root
|--train
|----class1
|----class2
|----...
|----classn
|--test
|----class1
|----class2
|----...
|----classn
|--valid
|----class1
|----class2
|----...
|----classn

#### Callbacks
The training period can be manually controlled.The patience the reference of lr 
reduction and early stopping can be modified by simply changing the parameters.

The tensorboard visualization backlog will be restored in './model'.Using instructions
'tensorboard --logdir=./model' can see visualization on the browser with address
'localhost:8080'(The port logout can be modified by annotation : --port=XXXX).

### Save model
The trained model can be save by 'model.save('model_name.h5')'.

### Testing
As the training set is already generated when we import the dataset,
the testing result is calculated by 
'Pred = model.evaluate_generator(testing_set, verbose=1)'
