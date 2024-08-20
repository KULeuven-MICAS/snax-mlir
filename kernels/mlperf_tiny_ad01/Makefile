include ../../runtime/Makefile.rules
include ../../runtime/snax-streamer-gemm.rules

ad01_int8.tflite:
	../../runtime/get_model.py anomaly_detection $@
	
SNAXPREPROCFLAGS = -p preprocess-mlperftiny --allow-unregistered-dialect

SNAXOPTFLAGS = -p dispatch-kernels,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions,insert-accfg-op{accelerator=snax_gemm},insert-accfg-op{accelerator=snax_simd},guarded-linalg-to-memref-stream,schedule-memref-linalg,memref-streamify,stream-snaxify,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma,memref-to-snax,snax-to-func,clear-memory-space

data.c data.h:
	$(PYTHON) gendata.py

%.x: %.o main.o data.o
	$(LD) $(LDFLAGS) $^ -o $@

sim_%: %
	rm -fr ./logs/
	$(VLTSIM) $<

ad01_int8.preproc3.mlir: ad01_int8.preproc2.mlir
	$(SNAXOPT) $(SNAXPREPROCFLAGS) -o $@ $<

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
