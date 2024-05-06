include ../../runtime/Makefile.rules

ad01_int8.tflite:
	../../runtime/get_model.py anomaly_detection $@

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
