CC 			=	gcc
CFLAGS		=	-O3 -Wunused-result
PROG		=	sw

all:$(PROG)

sw: sw.c
	$(CC) $(CFLAGS) $^ -o $@ 

.PHONY:clean
clean:
	rm -f $(PROG)