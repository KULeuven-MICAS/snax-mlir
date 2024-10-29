include ../../runtime/snax-gemmx.rules
include ../../runtime/Makefile.rules

ad01_int8.tflite:
	../../runtime/get_model.py anomaly_detection $@

ad01_int8.preprocfinal.mlir: ad01_int8.mlir
	$(SNAXOPT) -p preprocess-mlperftiny --print-op-generic --allow-unregistered-dialect -o $@ $<

SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,preprocess-mlperftiny-2,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},convert-stream-to-snax-stream#,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma{test_ignore_transform=true},memref-to-snax,snax-to-func,test-debug-to-func,convert-kernel-to-linalg,clear-memory-space
#AXOPTFLAGS = -p 'convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,reuse-memref-allocs,alloc-to-local,set-memory-space,realize-memref-casts,reuse-memref-allocs,set-memory-layout,insert-sync-barrier,pipeline,dispatch-regions{nb_cores=3},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma{test_ignore_transform=true},memref-to-snax,clear-memory-space,mlir-opt{generic=true arguments=-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic},snax-to-func{test_static_alloc=true},convert-kernel-to-linalg,test-rescale-to-trunc,linalg-to-library-call'

data.c data.h:
	$(PYTHON) gendata.py

%.x: %.o main.o data.o
	$(LD) $(LDFLAGS) $^ -o $@

sim_%: %
	rm -fr ./logs/
	$(VLTSIM) $<

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite
