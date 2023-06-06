CC 			=	nvcc
CFLAGS		=	-O3
PROG		=	sw-cuda

all:$(PROG)

sw-cuda: sw-cuda.cu
	$(CC) $(CFLAGS) $^ -o $@ 

.PHONY:clean
clean:
	rm -f $(PROG)