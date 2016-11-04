CC=gcc
CFLAGS=-O0 -g

LNK=$(CC)
LNKFLAGS=-lm

all: nn

nn.o : nn.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

nn : nn.o
	$(LNK) $(LNKFLAGS) $< -o $@

clean :
	@rm -f nn *.o
