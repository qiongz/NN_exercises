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
./test_mlr
<pre><code>
Cost of train/validation at epoch    0 :  0.45934263 0.34229076 
Cost of train/validation at epoch   10 :  0.25960219 0.28029829 
Cost of train/validation at epoch   20 :  0.24500903 0.27580458 
Cost of train/validation at epoch   30 :  0.23828110 0.27294943 
validation set accuracy:0.925119
Enter image id (0<=id<8400):<br/>
0
----------------------------




            *********       
           ***********      
           ************     
           ************     
                    ***     
                    ***     
                    ***     
                   ****     
                  *****     
                 *****      
                *****       
      **************        
    ***************         
   ****************         
   *****************        
   ******************       
   **********     ****      
   ********       *****     
    ****           ****     
                    ***     




----------------------------
Predicted: 2
Actual: 2

</code></pre>

### feed-forward neural network with two hidden-layers
Initialize the hidden layer dimensions with {256,128}<br/>
activation types with {"ReLU","sigmoid"}<br/>
keep probabilities in dropout regularization with {0.5,0.6}:<br/>
./test_ffnn
<pre><code>
Cost of train/validation at epoch    0 :  0.61099398 0.29974398 
Cost of train/validation at epoch   10 :  0.04274188 0.08710998 
Cost of train/validation at epoch   20 :  0.01536978 0.07854702 
Cost of train/validation at epoch   30 :  0.01044570 0.07815988 
validation set accuracy:0.977024
</code></pre>
The accuracy is better than the logistic regression

## License
[MIT](https://choosealicense.com/licenses/mit/)
