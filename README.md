rnn_ctc
=======

Recurrent Neural Network with Connectionist Temporal Classifical implemented in Theano. Includes a Toy training example.

1) First generate some data using scribe
```sh
python3 scribe.py data.pkl
```
This will output data.pkl

2) Run the actual recurrent neural net with connectionist temporal classification cost function as:
```sh
python3 rnn_ctc data.pkl
```

Sample Output
-------------
Input: 0 2 3 1 0 1 0 2 1 2   
 0¦░░░░██░░██░██░░░░░░░░░░░░░¦  
 1¦░░░░░███░░███░░███░░░░░░░░¦  
 2¦░░░░████░░░░████░░████░░░░¦  
 3¦░░░░█████░░░░░░░░░░░░░░░░░¦  

Predn: 0 2 3 1 0 1 0 2 1 2   
 0¦░░░░█░░░░██░█▓░░░░░░░░░░░░¦  
 1¦░░░░░░░██░░█░░░░███░░░░░░░¦  
 2¦░░░░░█░░░░░░░▒██░░░██▓░░░░¦  
 3¦░░░░░░█░░░░░░░░░░░░░░░░░░░¦  
 4¦████░░░░░░░░░░░░░░░░░▒████¦  

Credits
-------
Based on https://github.com/shawntan/rnn-experiment/
