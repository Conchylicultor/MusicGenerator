##

For this project, due to the lack of open sources python 3 compatibles library, I quickly implemented a simple higher level lib based on [mido](https://github.com/olemb/mido) to read/write midi files. It's really basic so don't expect to support the full midi specification but for simple songs, it's quite efficient. Here is an example to generate a new song:

```python
import deepmusic.songstruct as music
from deepmusic.midiconnector import MidiConnector

test_song = music.Song()
main_track = music.Track()

for i in range(44):  # Add some notes
    new_note = music.Note()

    new_note.note = (i%2)*(21+i) +((i+1)%2)*(108-i)
    new_note.duration = 32
    new_note.tick = 32*i  # Absolute time in tick from the begining
    
    main_track.notes.append(new_note)
 
test_song.tracks.append(main_track)
MidiConnector.write_song(test_song, 'data/midi/test.mid')
```
