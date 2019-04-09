ICC              := icpc
CFLAGS           := -std=c++11 
MKL_CFLAGS       := -m64 -I$(MKLROOT)/include  

PARAL_MKL_LIBS   := -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64  \
	  	    -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# ----------------------------------------
default:
	@echo "Available make targets are:"
	@echo "  make test_lr       # compiles logistic_regression.cpp nn_lr.cpp"

all:test_lr

test_lr:lr.o init.o utils.o nn_lr.o
	$(ICC) $^ -O3 -o $@  ${PARAL_MKL_LIBS}

lr.o:logistic_regression.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

init.o:init.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

utils.o:utils.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

nn_lr.o:nn_lr.cpp
	$(ICC) $(CFLAGS) $(MKL_CFLAGS) $^ -c -o $@

.PHONY:clean remove
clean:
	rm -f *.o 

remove:
	rm -f  test_lr


