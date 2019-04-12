# Neural network exercises
Learning neural networks by exercises.
## Prerequisties
Libraries:
* icc,mkl

## Installing
sudo apt-get install g++ g++-multilib build-essential<br/>
install icc,mkl<br/>
make test_lr test_fnn<br/>

## Datasets
datasets/train.csv, datasets/dev.csv

the training/validation data sets uses the MNIST datasets in csv format:<br/>
https://www.kaggle.com/c/digit-recognizer/data

## Running the test
### multi-classes logistic regression
./test_mlr
<pre><code>
Cost of train/validation at epoch    0 :  1.53400493 1.06833994
Cost of train/validation at epoch   50 :  0.32598802 0.32673740
Cost of train/validation at epoch  100 :  0.29563040 0.30209836
Cost of train/validation at epoch  150 :  0.28177088 0.29180923
Cost of train/validation at epoch  200 :  0.27304924 0.28599998
Cost of train/validation at epoch  250 :  0.26649177 0.28233498
Cost of train/validation at epoch  300 :  0.26153630 0.27988511
Cost of train/validation at epoch  350 :  0.25738254 0.27816206
Cost of train/validation at epoch  400 :  0.25402132 0.27682486
Cost of train/validation at epoch  450 :  0.25106934 0.27578330
validation set accuracy:0.923571
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
Initialize the hidden layer dimensions with {256,64}<br/>
activation types with {"ReLU","sigmoid"}<br/>
keep probabilities in dropout regularization with {0.7,0.8}:<br/>
./test_ffnn
<pre><code>
Cost of train/validation at epoch    0 :  2.26165628 2.17197156
Cost of train/validation at epoch   50 :  0.26327664 0.26657039
Cost of train/validation at epoch  100 :  0.17473789 0.18758176
Cost of train/validation at epoch  150 :  0.12650602 0.14836265
Cost of train/validation at epoch  200 :  0.09608501 0.12560201
Cost of train/validation at epoch  250 :  0.07490626 0.11114754
Cost of train/validation at epoch  300 :  0.05939516 0.10186721
Cost of train/validation at epoch  350 :  0.04772070 0.09572258
Cost of train/validation at epoch  400 :  0.03852429 0.09181294
Cost of train/validation at epoch  450 :  0.03158921 0.08901041
validation set accuracy:0.97369
</code></pre>
The accuracy is better than the logistic regression

## License
[MIT](https://choosealicense.com/licenses/mit/)
