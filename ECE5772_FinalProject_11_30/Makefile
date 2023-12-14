# Compiler/linker setup ------------------------------------------------------

# Mac OS X-specific flags.  Comment these out if using Linux.
#PLATFORM = osx
#CC       = gcc
#CFLAGS   = -fast -Wall
#OSLIBS   = -Wl,-framework -Wl,IOKit
#LDFLAGS  =

# Linux-specific flags.  Comment these out if using Mac OS X.
PLATFORM = linux
CC       = gcc -Wall
CPP		 = g++
STD11	 = -std=c++11
TBB      = -ltbb
PTHREAD  = -lpthread
MATH 	 = -lm
CFLAGS   = -O3 -Wall # can't include -lpthread here.. find out why
OSLIBS   =
LDFLAGS  =

# make all: all OBJS are compiled and executable available
# make <individual>
# make clean
# '-lpthread' has to be included at the end of the command (can't be in the CFLAGS)
# do not use spaces (tabs are allowed)
# Example programs -----------------------------------------------------------
OBJS = nnet simulate
all: $(OBJS)

nnet: nnet.cpp
	$(CPP) $(STD11) -o $@ $^ $(TBB)

simulate: simulate.cpp
	$(CPP) $(STD11) -o $@ $^ $(TBB)
# library


# Maintenance and stuff ------------------------------------------------------
clean:
	rm -f $(OBJS) *.o core