include ../../runtime/Makefile.rules

kws_ref_model.tflite:
	../../runtime/get_model.py keyword_spotting $@

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
