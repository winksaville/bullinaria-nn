CC=gcc
CFLAGS=-O2 -g

LNK=$(CC)
LNKFLAGS=-lm

OD=objdump

all: nn

nn.o : nn.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

nn : nn.o
	$(LNK) $(LNKFLAGS) $< -o $@

nn.txt : nn
	$(OD) -d $< > $@

clean :
	@rm -f nn *.o nn.txt
