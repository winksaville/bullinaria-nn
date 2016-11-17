# Parameters:
#   DBG=0 or 1 (default = 0)

# _DBG will be 0 if DBG isn't defined on the command line
_DBG = 0$(DBG)

# Make sure _EPOCH_COUNT is at least 1
_EPOCH_COUNT = +$(EPOCH_COUNT)
ifeq ($(_EPOCH_COUNT), +)
  _EPOCH_COUNT = 1
endif

CC=gcc
CFLAGS=-O2 -g -DDBG=$(_DBG) -DEPOCH_COUNT=$(_EPOCH_COUNT)

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
