ICC              := icpc
CFLAGS           := -std=c++11 
MKL_CFLAGS       := -I$(MKLROOT)/include  

MKL_LIBS         :=  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

MKL_PARAL_LIBS   := -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# ----------------------------------------
default:
	@echo "Available make targets are:"
	@echo "  make test_mlr test_ffnn test_conv   # compiles logistic_regression.cpp feed_forward_neural_network.cpp"

all:test_mlr test_ffnn test_conv clean

test_mlr:mlr.o init.o utils.o dnn.o layers.o
	$(ICC) $^ -O3 -o $@  ${MKL_LIBS}

test_ffnn:ffnn.o init.o utils.o dnn.o layers.o
	$(ICC) $^ -O3 -o $@  ${MKL_LIBS}

test_conv:conv.o init.o utils.o dnn.o layers.o
	$(ICC) $^ -O3 -o $@  ${MKL_LIBS}

conv.o:convolutional_neural_network.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

ffnn.o:feed_forward_neural_network.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

mlr.o:logistic_regression.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

init.o:init.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

utils.o:utils.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

dnn.o:dnn.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

layers.o:layers.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@
	
.PHONY:clean remove
clean:
	rm -f *.o 

remove:
	rm -f  test_mlr test_ffnn  test_conv


