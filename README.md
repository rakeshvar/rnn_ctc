# rnn_ctc

Recurrent Neural Network with Connectionist Temporal Classifical implemented 
in Theano. Includes a Toy training example.

## Usage

### Generate Data
First generate some data using one of the scribes (a, b or c)
```sh
# Run with '-h' to see full functionality of gen_data.py
python3 gen_data.py -h
```

a) Hindu numerals   
```sh
python3 gen_data.py data.pkl -a hindu
```

b) ASCII characters.   
```sh
python3 gen_data.py data.pkl -a ascii
```

c) scribe_rows scribes i-th digit along the i-th row as an i+2 long bar  
```sh
python3 alphabets/scribe_rows.py data.pkl
# Run with no arguments to see full usage
```

Now you have the data.pkl file.

### Train  Network
Run the actual Recurrent Neural Net with Connectionist Temporal Classification 
cost function as:
```sh
python3 train.py data.pkl [configuration_num]
# Run with no arguments for full usage.
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

## Dependencies
* Numpy
* Theano

Can easily port to python2 by adding lines like these where necessary:
``` python
from __future__ import print_function
```
