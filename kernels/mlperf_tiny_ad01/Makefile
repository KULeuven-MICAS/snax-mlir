include ../../runtime/Makefile.rules

ad01_int8.tflite:
	../../runtime/get_model.py anomaly_detection $@


%.preprocfinal.mlir: %.mlir
	$(SNAXOPT) -p preprocess-mlperftiny --print-op-generic --allow-unregistered-dialect -o $@ $<

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
