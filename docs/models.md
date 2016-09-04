# Models

Here are presented my experiments and the models I used.

## Basic RNN

As baseline, I tried a simple RNN model. Given a keyboard configuration, I try to predict the next one (this architecture is similar to the famous Char-RNN model). I formulate the prediction as a 88-binary classification problems: for each note on the keyboard, the network try to guess if the note is pressed or released. Because all classifications are not mutually exclusive (two keys can be pressed at the same time), I use a sigmoid cross entropy instead of softmax. For this first try, a lot of assumptions have been made on the song (only 4/4 for signature, quarter-note as maximum note resolution, no tempo changes or other difficulties that the model could not handle).

At first, I try to trained this model on 3 Scott Joplin songs. I choose Ragtime music for the first test because Ragtime songs have a really rigid and well defined structure, and the songs satisfied the assumptions made above. Each song is slitted in small parts and I randomly shuffle the parts between the songs so the network learn simultaneously different songs.

Because there is a high bias toward the negative class (key not pressed), I was a little afraid that the network would only predict empty songs. It doesn't seems to be the case. On the contrary, the network clearly overfit. When analysing the generated song, we see that the network has memorized entire parts of the original songs. This simple model was mainly a test to validate the fact that a small/medium sized network can encode some rhythmic and harmonic information. Of course, the database wasn't big enough to have a truly original artificial composer, but I'm quite impress by the capability of the network to learn by heart the song.

Usually the first notes are quite randoms but after few bars, the network stabilize over a part of the song he remember. When trying to generate sequences longer than the training ones, the network will simply loop over some parts and play it indefinitely. You can listen one of the generated song [here](midi/basic_rnn_joplin.mid). We can clearly recognize parts of [Rag Dance](https://youtu.be/tCrj1s1iVas).
