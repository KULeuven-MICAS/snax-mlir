include ../../runtime/Makefile.rules

pretrainedResnet_quant.tflite:
	../../runtime/get_model.py image_classification $@

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
