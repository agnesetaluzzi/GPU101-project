CC 			=	gcc
CFLAGS		=	-O3 -Wunused-result
PROG		=	smith-waterman/sw-test

all:$(PROG)

smith-waterman/sw-test: smith-waterman/sw.c
	$(CC) $(CFLAGS) $^ -o $@ 

.PHONY:clean
clean:
	rm -f $(PROG)