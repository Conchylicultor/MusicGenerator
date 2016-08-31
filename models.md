Just a list of ideas or model to test

* From a bars/lyrical phrases, predict the next one
* Use 2 networks or more (idea from [Andy T](https://github.com/aPToul/Experiments-in-Music)): one to generate short melodies, one for the global structure.
* Maybe instead try some kind of inception architecture but for RNN (multiples RNN with different parameters (LTSM/GRU, hidden size, Relu/...) trained simultaneosly)
* Apply GAN (or its variant) to music generation
* Test with/without attention
* Recurent DBN (compress information in the middle lstm layers): something like [500,250,125,250,500] for the hidden layers
* Include somehow CRF or bidirectional LSTM ?
* Use 1d convolution NN on 1d grid input (Mixed with RNN): relative distance between notes is more important than absolute position. Pb: how to incorporate relation between frames (look at what has been done for video)
* Maybe instead use a 2d convolution on 2d grid input, reformulate the musical composition as an image generation problem (with the help of adversarial model)

* Training task neural art for music: play pieces with certain pattern/style.


What note representation ?

* Simpler model could be to have a single vector as input of 88 (keys) for each 1/16 of bars: the vector contains 0 if nothing has been played or 1 if the key has been pressed (note duration not taken into account). The song is represented by a giant matrix: (88*(16*nbBars)). This representation will be referred as grid input (1d or 2d if we add the temporal dimension to the input tensor).
* Artificially increase dataset: transpose musical pieces ?
* Midi vs ABC representation: one network just print the basic melody in ABC notation (text file as it was a char-rnn), the second network play this file as if it was improvising, playing chords and melody on some based tablature (as jazz man do).
* Maybe try something closer of the musical theory (1rst, 2nd, 3rd degree instead of A,D,C) or something closer to the physic (frequency ? relative distance behind the note). Somehow the model should understand that 2 notes like C3 and C4 "feel" the sames.
* Note2vec ? something which convert each note/chord into a multidimentional space (How to divide/separate chords/notes ? Pb of multiples representations: a same chord can be played in arpegio or using more complex partern. Should be a multilevel representation)
