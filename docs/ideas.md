Just an unorganised list of ideas or models to test.

* From a bars/lyrical phrases, predict the next one
* Use 2 networks or more (idea from [Andy T](https://github.com/aPToul/Experiments-in-Music)): one to generate short melodies, one for the global structure.
* Maybe instead try some kind of inception architecture but for RNN (multiples RNN with different parameters (LTSM/GRU, hidden size, Relu/...) trained simultaneosly)
* Apply GAN (or its variant) to music generation
* Test with/without attention
* Recurrent DBN (compress information in the middle lstm layers): something like \[500,250,125,250,500\] for the hidden layers
* Include somehow CRF or bidirectional LSTM ?
* Use 1d convolution NN on 1d grid input (Mixed with RNN): relative distance between notes is more important than absolute position. Pb: how to incorporate relation between frames (look at what has been done for video)
* Maybe instead use a 2d convolution on 2d grid input, reformulate the musical composition as an image generation problem (with the help of adversarial model)
* Also include the tempo change as prediction. Continuously (each time steps) or as event (Always predict 0 except sometimes) ? Prediction as a multi-class classification among some predetermined class (allegro, andante,...). Multi objective function with softmax (Loss=a*LossTempo + b*LossNotes).
* Try Skip-Thought Vectors, bidirectional-RNN (predict both past/future)
* Encoder/decoder network: The encoder encode the keyboard disposition at time t and the decoder predict at time t+1. All this network is a cell of the global RNN network. Pb: the notes are read sequencially (solving with bidirectional RNN ?)

* Training task neural art for music: play pieces with certain pattern/style.


What note representation ?

* Simpler model could be to have a single vector as input of 88 (keys) for each 1/16 of bars: the vector contains 0 if nothing has been played or 1 if the key has been pressed (note duration not taken into account). The song is represented by a giant matrix: (88\*(16\*nbBars)). This representation will be referred as grid input (1d or 2d if we add the temporal dimension to the input tensor).
* Artificially increase dataset: transpose musical pieces ?
* Midi vs ABC representation: one network just print the basic melody in ABC notation (text file as it was a char-rnn), the second network play this file as if it was improvising, playing chords and melody on some based tablature (as jazz man do).
* Maybe try something closer of the musical theory (1rst, 2nd, 3rd degree instead of A,D,C) or something closer to the physic (frequency ? relative distance behind the note). Somehow the model should understand that 2 notes like C3 and C4 "feel" the sames.
* Note2vec ? something which convert each note/chord into a multidimentional space (How to divide/separate chords/notes ? Pb of multiples representations: a same chord can be played in arpegio or using more complex partern. Should be a multilevel representation)



Train conjointly recurrent CNN  for the spacial dependency (use lstm containing a CNN ?) AND a standard RNN for the absolute position (best of both words ?) At the end a fully connected layer mix the two outputs to produce the final result: the CNN provide the pattern and the standard network provider the position (use simple res net)

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
A first simple solution could be to try to optimize 2 task conjointly (the binary classification on all keyboards key and one on the notes % 12). Pb is: How to compute the prediction for the %12 from the global keyboard prediction ? A simple way could be to simply add the prediction of each note (P(C)=sigm(C1 + C2 + C3 +...))



TODO

Plot piano roll image while training with some samples of the train and train set (every 1000 iterations). Saved on a subfolder (training/) ?

Try learning on multiple sample length at the same time (short, longer)

OpenCv
When testing, plot the prediction color map and the ground truth conjointly. Do it for training/testing/generatives songs

Apply k neighbors to find similar segment in the dataset

Include sample length dans le titre des sons predits?





Notes should be modulo 12 (no distinction between C3 and C5).
Limitation of CNN at the boundaries (solution: try cycling: copy notes at the boundaries. Slide the kernel until it has done a cycle). Try 12*4 kernels for the CNN > contains chords

CNN is here to learn chords/patterns

Use a 2d cnn one dimension for the chords, the other for the pattern (alberti bass); or use a RNN for the pattern instead

If a bar is 4 tics. Divide create 4 images for each one of the tics ?
Other solution is use the cnn/RNN structure the CNN has a temporal resolution of 2 tics for instance. The RNN part has a temporal resolution of 1/4 of tics. That's mean at each RNN step, the network re-get some information he has seen on previous step.

One of the output of the neural networks control the bpm. The cnn/network itself don't take care of the speed (1 tic is 1 tic) but when playing, the network can send signals to increase/decrease the tempo (maybe 5 prebuild tempo and predition as classification problem though softmax).

Tracks as channel ??

Instead of randomly splitting the tracks, only split when there is a bar !! With that, the prediction will be synchronised add probably more 'clean'

Try shorter sequences (4, 8, 16 ?) For the sample length

Try tu train the network with variables length sentences

Having a 2 way RNN network. At each timestep, first, the network slide over the notes to encode the entire keyboard configuration. A decoder predict the next configuration. The decoder output and a second state vector are connected to the next encoder. Should be difficult to optimize due to the deep of the network (88*sample_length steps for one timestep). 
Try to use some synthetic gradient tricks ? We backpropagate on each timestep independently. The state vector of the previous step is a learning parameter (how to connect both sides !???!)

GAN: Use RNN to generate piano roll(image). Then Cnn to discriminate

Select sample in function of the length of the song and samples sequences (otherwise bias towards short songs): something like nb_sample = 2*song_length//sample_ length

Load starter sequence for testing (read initial notes from files): input sequence given from which the network will predict what's comes next.

Control loop fct with List\[boolean placeholder\] (use_previous)

Keep track of ratio update/weights
Keep track of magnitude of some values (internal state of rnn, weights...)
magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory\[1\])))
tf.scalar_summary("magnitude at t=1", magnitude)

tf.train.GradientDescentOptimizer class (and related Optimizer classes) call tf.gradients() internally. If you want to access the gradients that are computed for the optimizer, you can call optimizer.compute_gradients() and optimizer.apply_gradients() manually, instead of calling optimizer.minimize()


How to monitor dead unit relu ??

Try peephole with lstm, apparently better timing (*Learning Precise Timing with LSTM Recurrent Networks*, ... et al.)

Try rnn pixel like: predict P(xt|xt-1,…,h)
Instead of taking a vector corresponding to a keyboard configuration, each input is just a note (pitch, distance from previous)

Try changing the note representation: instead of a 88*1, do a 12*nbOctave


## Tools

Some python library for midi file manipulation
* Mido: seems the best one now for low level manipulation. Close the the original specs (used in this program).
* pretty_midi: higher level (piano rolls function could be really handy). Not python 3 compatible, had to program myself the piano roll conversion.
