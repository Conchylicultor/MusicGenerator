# Models

Here are presented my experiments and the models I used.

## Basic RNN

As baseline, I tried a simple RNN model. Given a keyboard configuration, I try to predict the next one. I formulate the prediction as a multi-binary classification problem: for each note on the keyboard, the network try to guess if the note is pressed or released.

I trained this model on Scott Joplin songs. I choose Ragtime music for the first test because Ragtime songs have a really rigid and well defined structure. There is no change in tempo or other difficulties that the model could not handle.
