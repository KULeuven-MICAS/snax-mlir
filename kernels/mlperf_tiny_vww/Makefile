include ../../runtime/Makefile.rules

vww_96_int8.tflite:
	../../runtime/get_model.py visual_wake_words $@

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
