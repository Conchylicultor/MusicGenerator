Just a list of ideas or model to test

* From a bars/lyrical phrases, predict the next one
* Use 2 networks or more (idea from [Andy T](https://github.com/aPToul/Experiments-in-Music)): one to generate short melodies, one for the global structure
* Apply GAN (or its variant) to music generation
* Test with/without attention

What note representation ?

* Midi vs ABC representation: one network just print the basic melody in ABC notation (text file as it was a char-rnn), the second network play this file as if it was improvising, playing chords and melody on some based tablature (as jazz man do).
* Maybe try something closer of the musical theory (1rst, 2nd, 3rd degree instead of A,D,C) or something closer to the physic (frequency ? relative distance behind the note). Somehow the model should understand that 2 notes like C3 and C4 "feel" the sames.
* Note2vec ? something which convert each note/chord into a multidimentional space (How to divide/separate chords/notes ? Pb of multiples representations: a same chord can be played in arpegio or using more complex partern. Should be a multilevel representation)
