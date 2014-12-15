# rnn_ctc

Recurrent Neural Network with Connectionist Temporal Classifical implemented in Theano. Includes a Toy training example.

## Usage

1) First generate some data using one of the scribes (a or b)

a) scribe_rows scribes i-th digit along the i-th row as an i+2 long bar  
```sh
python3 scribe_rows.py data.pkl
# Run with no arguments to see full usage
```

b) scribe_hindu scribes i-th digit as a Hindu-Arabic numeral.   
```sh
python3 scribe_hindu.py data.pkl
# Run with no arguments to see full usage
```
This will output data.pkl

2) Run the actual Recurrent Neural Net with Connectionist Temporal Classification cost function as:
```sh
python3 rnn_ctc data.pkl [nHidden]
```

## Sample Output
```
# Using data from scribe_rows.py
Shown : 0 2 3 1 0 1 0 2 1 2   
Seen  : 0 2 3 1 0 1 0 2 1 2   
Images (Shown & Seen) : 

 0¦    ██  ██ ██             ¦  
 1¦     ███  ███  ███        ¦  
 2¦    ████    ████  ████    ¦  
 3¦    █████                 ¦  

 0¦    █    ██ █▓            ¦  
 1¦       ██  █    ███       ¦  
 2¦     █       ▒██   ██▓    ¦  
 3¦      █                   ¦  
 4¦████                 ▒████¦  

# Using data from scribe_hindu.py
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

## Credits
* Graves, Alex, et al. ["Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks."](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf) ICML, 2006.
* Theano implementation of CTC by [Shawn Tan](https://github.com/shawntan/rnn-experiment/)

## Dependencies
* Numpy
* Theano

Can easily port to python2 by adding the following line where necessary:
``` python
from __future__ import print_function
```
