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
keep probabilities in dropout regularization with {0.9,0.6,0.8}:<br/>
./test_ffnn -n 100 <br/>
<pre><code>
Cost of train/validation at epoch    0 :  0.88942140 0.34440878 
Cost of train/validation at epoch   10 :  0.16244245 0.10217448 
Cost of train/validation at epoch   20 :  0.10956403 0.08221784 
Cost of train/validation at epoch   30 :  0.09188860 0.06962664 
Cost of train/validation at epoch   40 :  0.07479116 0.06810240 
Cost of train/validation at epoch   50 :  0.06701186 0.06320439 
Cost of train/validation at epoch   60 :  0.06487222 0.06216515 
Cost of train/validation at epoch   70 :  0.05761564 0.06120869 
Cost of train/validation at epoch   80 :  0.05512367 0.05802767 
Cost of train/validation at epoch   90 :  0.05044307 0.05985889 
Cost of train/validation at epoch  100 :  0.05259707 0.05880265 
validation set accuracy:0.982143
</code></pre>
The accuracy is much better than the logistic regression

## License
[MIT](https://choosealicense.com/licenses/mit/)
