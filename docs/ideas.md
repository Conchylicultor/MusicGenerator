Just an unorganised list of ideas or models to test.

* From a bars/lyrical phrases, predict the next one
* Use 2 networks or more (idea from [Andy T](https://github.com/aPToul/Experiments-in-Music)): one to generate short melodies, one for the global structure.
* Maybe instead try some kind of inception architecture but for RNN (multiples RNN with different parameters (LTSM/GRU, hidden size, Relu/...) trained simultaneosly)
* Apply GAN (or its variant) to music generation
* Test with/without attention
* Recurrent DBN (compress information in the middle lstm layers): something like [500,250,125,250,500] for the hidden layers
* Include somehow CRF or bidirectional LSTM ?
* Use 1d convolution NN on 1d grid input (Mixed with RNN): relative distance between notes is more important than absolute position. Pb: how to incorporate relation between frames (look at what has been done for video)
* Maybe instead use a 2d convolution on 2d grid input, reformulate the musical composition as an image generation problem (with the help of adversarial model)
* Also include the tempo change as prediction. Continuously (each time steps) or as event (Always predict 0 except sometimes) ? Prediction as a multi-class classification among some predetermined class (allegro, andante,...). Multi objective function with softmax (Loss=a*LossTempo + b*LossNotes).

* Training task neural art for music: play pieces with certain pattern/style.


What note representation ?

* Simpler model could be to have a single vector as input of 88 (keys) for each 1/16 of bars: the vector contains 0 if nothing has been played or 1 if the key has been pressed (note duration not taken into account). The song is represented by a giant matrix: (88*(16*nbBars)). This representation will be referred as grid input (1d or 2d if we add the temporal dimension to the input tensor).
* Artificially increase dataset: transpose musical pieces ?
* Midi vs ABC representation: one network just print the basic melody in ABC notation (text file as it was a char-rnn), the second network play this file as if it was improvising, playing chords and melody on some based tablature (as jazz man do).
* Maybe try something closer of the musical theory (1rst, 2nd, 3rd degree instead of A,D,C) or something closer to the physic (frequency ? relative distance behind the note). Somehow the model should understand that 2 notes like C3 and C4 "feel" the sames.
* Note2vec ? something which convert each note/chord into a multidimentional space (How to divide/separate chords/notes ? Pb of multiples representations: a same chord can be played in arpegio or using more complex partern. Should be a multilevel representation)



Train conjointly récurent CNN  for the spacial dependency (use lstm containing a CNN ?) AND a standard RNN for the absolute position (best of both words ?) At the end a fully connected layer mix the two outputs to produce the final result: the CNN provide the pattern and the standard network provider the position (use simple res net)

Look at deconvolution
ResNet

Input grid 2d: use more channels to represent velocity/duration ?
Or simply the value of the cell represent the duration instead of binary

Training: Play one bars and only after start to backpropagate for each steps (no need to penalize when the network has no way to know if he is doing right or wrong, meaning at the beginning). The idea is that the first bar is given for free just as initiation/setup (prendre de l'élant).
For training, force the correct answer for each timestep

Visualize the filters of the CNN.

Walkthrough:

* At first try simple midi (Joplin, pop, Bach, Mozart). Then try more complex composer (Chopin, Rachmaninoff < difficult for the change of tempo within the play)


What cost fct ?

Eventually, the penalty should be less if the network predict the same note but not in the right pitch (ex: C4 instead of C5), with a decay the further the prediction is (D5 and D1 more penalized than D4 and D3 if the target is D2)


## Tools

Some python library for midi file manipulation
* Mido: seems the best one now for low level manipulation. Close the the original specs.
* pretty_midi: higher level (piano rolls function could be really handy)
