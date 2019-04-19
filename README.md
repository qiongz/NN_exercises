# Neural network exercises
Learning neural networks by exercises.
## Prerequisties
Libraries:
* icc,mkl

## Installing
sudo apt-get install g++ g++-multilib build-essential<br/>
install icc,mkl<br/>
make test_mlr test_ffnn<br/>

## Datasets
datasets/train.csv, datasets/test.csv

the training/validation data sets uses the MNIST datasets in csv format:<br/>
https://www.kaggle.com/c/digit-recognizer/data

## Running the test
### multi-classes logistic regression
Using dropout with keep_prob=0.85 to perform regularization in the input layer <br/>
./test_mlr -n 100 -a 0.003
<pre><code>
Cost of train/validation at epoch    0 :  0.58074164 0.35054046 
Cost of train/validation at epoch   10 :  0.27973989 0.26971540 
Cost of train/validation at epoch   20 :  0.27052698 0.27189568 
Cost of train/validation at epoch   30 :  0.26301974 0.27001908 
Cost of train/validation at epoch   40 :  0.25950569 0.26645681 
Cost of train/validation at epoch   50 :  0.25770107 0.26931667 
Cost of train/validation at epoch   60 :  0.26052868 0.27073887 
Cost of train/validation at epoch   70 :  0.25515199 0.26677442 
Cost of train/validation at epoch   80 :  0.25009876 0.26658201 
Cost of train/validation at epoch   90 :  0.24885687 0.26687291 
Cost of train/validation at epoch  100 :  0.24961139 0.26613927 
validation set accuracy:0.928571
</code></pre>

### feed-forward neural network with two hidden-layers
Initialize the hidden layer dimensions with {768,512}<br/>
activation types with {"ReLU","ReLU"}<br/>
keep probabilities in dropout regularization with {0.85,0.4,0.5}:<br/>
./test_ffnn -n 50 -a 0.001 <br/>
<pre><code>
Cost of train/validation at epoch    0 :  0.75169951 0.24952073 
Cost of train/validation at epoch   10 :  0.11169609 0.08198518 
Cost of train/validation at epoch   20 :  0.06751236 0.07157799 
Cost of train/validation at epoch   30 :  0.04579250 0.06736103 
Cost of train/validation at epoch   40 :  0.03172652 0.06837747 
Cost of train/validation at epoch   50 :  0.02747416 0.06784185 
validation set accuracy:0.983095
</code></pre>
The accuracy is much better than the logistic regression

### convolutional neural network 

The test initialize the convolutional network with three convolutional and pooling layers <br/>
Input layer: dropout keep_prob=0.9,<br/>
1st Conv2d layer: {filter_size=3, padding=1, stride=1, n_channel=16} <br/>
1st Pool layer: {filter_size=2,stride=2,n_channel=16}  <br/>
2nd Conv2d layer: {filter_size=3, padding=1, stride=1, n_channel=32} <br/>
2nd Pool layer: {filter_size=2,stride=2,n_channel=32} <br/>
3rd Conv2d layer: {filter_size=3, padding=1, stride=1, n_channel=64} <br/>
3rd Pool layer: {filter_size=2,stride=2,n_channel=64} <br/>
1st Hidden layer: dim=512, dropout keep_prob=0.5 <br/>
2nd Hidden layer: dim=256, dropout keep_prob=0.6 <br/>

using batch_size=128 and Adam optimization with learning rate decay:

./test_conv -n 10 -a 0.0025 <br/>
<pre><code>
Cost of train/validation at epoch    0 :  0.51284289 0.12400043 
Cost of train/validation at epoch    1 :  0.10023427 0.06492750 
Cost of train/validation at epoch    2 :  0.06424299 0.04772324 
Cost of train/validation at epoch    3 :  0.04936956 0.04153788 
Cost of train/validation at epoch    4 :  0.03802959 0.04588846 
Cost of train/validation at epoch    5 :  0.03091952 0.03321058 
Cost of train/validation at epoch    6 :  0.02083942 0.03290170 
Cost of train/validation at epoch    7 :  0.01673818 0.03719120 
Cost of train/validation at epoch    8 :  0.01175991 0.03252161 
Cost of train/validation at epoch    9 :  0.00862098 0.03288486 
Cost of train/validation at epoch   10 :  0.00722232 0.03297896 
validation set accuracy:0.992024
</code></pre>
The accuracy improves, while we could see that we're still over-fitting, batch-normalization will be updated in the further version.

## License
[MIT](https://choosealicense.com/licenses/mit/)
