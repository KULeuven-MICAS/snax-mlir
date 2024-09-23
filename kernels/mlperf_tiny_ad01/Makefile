include ../../runtime/snax-gemmx.rules
include ../../runtime/Makefile.rules

ad01_int8.tflite:
	../../runtime/get_model.py anomaly_detection $@

ad01_int8.preprocfinal.mlir: ad01_int8.mlir
	$(SNAXOPT) -p preprocess-mlperftiny --print-op-generic --allow-unregistered-dialect -o $@ $<

SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,test-insert-debugs,set-memory-space,set-memory-layout,test-insert-debugs,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},guarded-linalg-to-memref-stream,schedule-memref-linalg,stream-snaxify,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,test-debug-to-func,convert-kernel-to-linalg,clear-memory-space

data.c data.h:
	$(PYTHON) gendata.py

%.x: %.o main.o data.o
	$(LD) $(LDFLAGS) $^ -o $@

sim_%: %
	rm -fr ./logs/
	$(VLTSIM) $<

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
