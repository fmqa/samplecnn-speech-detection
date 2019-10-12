PYTHON ?= python
WGET ?= wget
TAR ?= tar
CP ?= cp
RM ?= rm -f
LIMIT ?= 4000

export LIMIT

model-gztan-speech-music-$(LIMIT).pth model-gztan-speech-music-$(LIMIT).log: samplecnn.py | music_speech
	$(PYTHON) -Bu samplecnn.py 2>&1 >>model-gztan-speech-music-$(LIMIT).log
	$(CP) model-gztan-speech-music.pth model-gztan-speech-music-$(LIMIT).pth

music_speech: music_speech.tar.gz
	[ -d $@ ] || $(TAR) --mode=755 -xf music_speech.tar.gz

music_speech.tar.gz:
	$(WGET) -O $@ http://opihi.cs.uvic.ca/sound/$@

.PHONY: clean-model clean distclean

clean-model:
	$(RM) -r model-gztan-speech-music.pth opt-gztan-speech-music.pth sched-gztan-speech-music.pth batches-gztan-speech-music.txt

clean: clean-model
	$(RM) -r music_speech

distclean: clean
	$(RM) -r music_speech.tar.gz

