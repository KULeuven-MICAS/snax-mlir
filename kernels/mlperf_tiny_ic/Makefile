include ../../runtime/snax-gemmx.rules
include ../../runtime/Makefile.rules

#  pretrainedResnet_quant.tflite:
#  	../../runtime/get_model.py image_classification $@


pretrainedResnet_quant.preprocfinal.mlir: pretrainedResnet_quant.mlir
	$(SNAXOPT) -p preprocess-mlperftiny --print-op-generic --allow-unregistered-dialect -o $@ $<


SNAXOPTFLAGS = -p convert-linalg-to-kernel,insert-accfg-op{accelerator=snax_gemmx},dispatch-kernels,convert-linalg-to-stream,fuse-streaming-regions,stream-bufferize,snax-bufferize,preprocess-mlperftiny-2,set-memory-space,set-memory-layout,realize-memref-casts,insert-sync-barrier,dispatch-regions{nb_cores=3},convert-stream-to-snax-stream,convert-linalg-to-accfg,convert-accfg-to-csr,snax-copy-to-dma{test_ignore_transform=true},memref-to-snax,snax-to-func,test-debug-to-func,convert-kernel-to-linalg,clear-memory-space,test-rescale-to-trunc


pretrainedResnet_quant.snax-opt.mlir: pretrainedResnet_quant.preprocfinal.mlir
	$(SNAXOPT) $(SNAXOPTFLAGS) --print-op-generic --allow-unregistered-dialect -o $@ $<

clean:
	rm -fr *.ll12 *.x *.o *.logs/ logs/ *.tflite

data.c data.h:
	$(PYTHON) gendata.py

%.x: %.o main.o data.o
	$(LD) $(LDFLAGS) $^ -o $@

sim_%: %
	rm -fr ./logs/
	$(VLTSIM) $<
