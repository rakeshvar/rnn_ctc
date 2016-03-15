# RNN CTC

Recurrent Neural Network with Connectionist Temporal Classifical implemented 
in Theano. Includes toy training examples.

## Use

The goal of this problem is to train a Neural Network (with recurrent connections) to learn to read 
sequences. As a part of the training we show it a series of such sequences (tablets of text in 
our examples) and also tell it what the tablet contains  (the labels of the written characters). 
 
## Methodology

We need to keep feeding our RNN the samples of text in two forms (written and labelled). If you 
have your own written samples you can train our system the **offline** way. If you have a 
*scribe* that can generate samples as you go, you can train one sample at a time, 
the **online** way. 

## Specifying parameters

You will need to specify a lot of parameters. Here is a overview. The file `configs/default.ast` 
has all the parameters specified (as a python dictionary), so compare that with these instructions.

* Data Generation (cf. `configs/alphabets.ast`)
    * Scribe (The class that generates the samples)
        * `alphabet`: 'ascii_alphabet' (0-9a-zA-Z etc.) or 'hindu_alphabet' (0-9 hindu numerals)
        * `noise`: Amount of noise in the image
        * `vbuffer`, `hbuffer`: horizontal and vertical buffers
        * `avg_seq_len`: Average length of the tablet  
        * `varying_len`: (bool) Make the length random
        * `nchars_per_sample`: This will make each tablet have the same number of characters. This 
        over-rides `avg_seq_len`.
    * `num_samples`

* Training (cf. `configs/default.ast`)
    * `num_epochs`
        * Offline case: Goes over the same data `num_epochs` times.
        * Online case: Each epoch has different data, resulting in generating a total of 
        `num_epochs * num_samples` unique data samples!
    * `train_on_fraction`
        * Offline case: Fraction of samples that are used as training data
        
* Neural Network (cf. `configs/midlayer.ast` and `configs/optimizers.ast`)
    * `use_log_space`: Perform calculations via the logarithms of probabilities.
    * `mid_layer`: The middle layer to be used. See the `nnet/layers` module for all the options you have.
    * `mid_layer_args`: The arguments needed for the middle layer. Depends on the `mid_layer`. 
    See the constructor of the corresponding `mid_layer` class. 
    * `optimizer`: The optimization algorithm to be used. `sgd`, `adagrad`, `rmsprop`, 
    `adadelta` etc. 
    * `optimzier_args`: The arguments that the optimizer needs. See the corresponding function in
     the file `nnet/updates.py`. 
        Note: This should **not** contain the learning rate.
    * `learning_rate_args`: 
        * `initial_rate`: Initial learning rate.
        * `anneal`: 
            * `constant`: Learning rate will be kept constant
            * `inverse`: Will decay as the inverse of the epoch.
            * `inverse_sqrt`: Will decay as the inverse of the square root of the epoch.
        * `epochs_to_half`: Rate at which the learning_rate is annealed. Higher number means 
        slower rate.

## Usage

### Offline Training
  
For this you need to generate data first and then train it using `train_offline.py`. 

##### Generate Data
You can use *hindu numerals* or the entire *ascii* set, specified via an ast file.

```sh
python3 gen_data.py <output_name.pkl> [config=configs/default.ast]*
```

##### Train  Network
You can train on the generated pickle file as:

```sh
python3 train_offline.py data.pkl [config=configs/default.ast]*
```

### Online Training
You can generate and train simultaneously as:

```sh
python3 train_online.py [config=configs/default.ast]*
```

## Examples

All the programs mentioned above can take multiple config files, later files override former ones.
 `configs/default.ast` is loaded by default.  

### Offline
```sh
# First generate the ast files based on given examples then...
python3 gen_data.py hindu_avg_len_60.py configs/hindu.ast configs/len_60.ast
python3 train_offline.py hindu_3chars.py configs/adagrad.ast configs/bilstm.ast configs/ilr.01.ast
```

### Online
```sh
python3 train_online.py configs/hindu.ast configs/adagrad.ast configs/bilstm.ast configs/ilr.01.ast
```

### Working Example
```sh
# Offline
python3 gen_data.py hindu3.py configs/working_eg.ast
python3 train_offline.py hindu3.py configs/working_eg.ast
# Online
python3 train_online.py configs/working_eg.ast
```


#Offline


## Sample Output
```
# Using data from scribe.py hindu
Shown : 0 2 2 5 
Seen  : 0 2 2 5 
Images (Shown & Seen) : 

 0¦                            ¦
 1¦          ██  ██            ¦
 2¦         █  ██  ████        ¦
 3¦           █   █ █          ¦
 4¦      ██  █   █  ███        ¦
 5¦     █  █████████  █        ¦
 6¦     █  █        █ █        ¦
 7¦      ██         ███        ¦
 
 0¦░░░░░░░░░█░░░░░░░░░░░░░░░░░░¦
 1¦░░░░░░░░░░░░░░░░░░░░░░░░░░░░¦
 2¦░░░░░░░░░░░░░█░░░█░░░░░░░░░░¦
 3¦░░░░░░░░░░░░░░░░░░░░░░░░░░░░¦
 4¦░░░░░░░░░░░░░░░░░░░░░░░░░░░░¦
 5¦░░░░░░░░░░░░░░░░░░░█▓░░░░░░░¦
 6¦█████████░███░███░█░▒███████¦

```
## References
* Graves, Alex. **Supervised Sequence Labelling with Recurrent Neural Networks.** Chapters 2, 3, 7 and 9.
 * Available at [Springer](http://www.springer.com/engineering/computational+intelligence+and+complexity/book/978-3-642-24796-5)
 * [University Edition](http://link.springer.com/book/10.1007%2F978-3-642-24797-2) via. Springer Link.
 * Free [Preprint](http://www.cs.toronto.edu/~graves/preprint.pdf)

## Credits
* Theano implementation of CTC by [Shawn Tan](https://github.com/shawntan/theano-ctc/)
* Updates.py from [Lasagne](http://lasagne.readthedocs.org/en/latest/modules/updates.html)

## Dependencies
* Numpy
* Theano

Can easily port to python2 by adding lines like these where necessary. In the interest of the 
future generations, we highly recommend you do not do that.
``` python
from __future__ import print_function
```
