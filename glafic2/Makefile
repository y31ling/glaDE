UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# 自动检测或使用环境变量设置的GLAFIC_ROOT
# 可以通过以下方式使用:
#   make GLAFIC_ROOT=/path/to/install
# 或者
#   export GLAFIC_ROOT=/path/to/install && make
ifndef GLAFIC_ROOT
	# 如果没有设置GLAFIC_ROOT，尝试自动检测
	MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
	GLAFIC_ROOT := $(shell dirname $(MAKEFILE_DIR))
endif

ifeq ($(UNAME_S),Darwin)
	ifeq ($(UNAME_M),x86_64)
		CFLAGS2 = -lcurl
		LIBPATH = /usr/local/lib
		INCPATH = /usr/local/include
		PY_LIBS = -lm -lcfitsio -lfftw3 -lgsl -lgslcblas
	else
		CFLAGS2 = -lcurl -lz
		LIBPATH = /opt/homebrew/lib
		INCPATH = /opt/homebrew/include
		PY_LIBS = -lm -lcfitsio -lfftw3 -lgsl -lgslcblas
	endif
else
	ifeq ($(UNAME_S),Linux)
		CFLAGS2 = 
		# 使用自动检测的或环境变量设置的路径
		LIBPATH = $(GLAFIC_ROOT)
		INCPATH = $(LIBPATH)/cfitsio-4.6.2:$(LIBPATH)/fftw-3.3.10/api:$(LIBPATH)/gsl-2.8
		PY_LIBS = -lm $(LIBPATH)/cfitsio-4.6.2/.libs/libcfitsio.so $(LIBPATH)/fftw-3.3.10/.libs/libfftw3.so $(LIBPATH)/gsl-2.8/.libs/libgsl.so $(LIBPATH)/gsl-2.8/cblas/.libs/libgslcblas.so -lcurl -lz -Wl,-rpath,$(LIBPATH)/gsl-2.8/.libs -Wl,-rpath,$(LIBPATH)/gsl-2.8/cblas/.libs -Wl,-rpath,$(LIBPATH)/cfitsio-4.6.2/.libs -Wl,-rpath,$(LIBPATH)/fftw-3.3.10/.libs
	else
		CFLAGS2 = 
		LIBPATH =
		INCPATH =
		PY_LIBS = 
	endif
endif

CC	= gcc
CFLAGS	= -O2 -Wall -fPIC -I$(LIBPATH)/cfitsio-4.6.2 -I$(LIBPATH)/fftw-3.3.10/api -I$(LIBPATH)/gsl-2.8
#LIBS    = -lm -lcfitsio -lfftw3 -lgsl -lgslcblas
LIBS	= -lm $(LIBPATH)/cfitsio-4.6.2/.libs/libcfitsio.a $(LIBPATH)/fftw-3.3.10/.libs/libfftw3.a $(LIBPATH)/gsl-2.8/.libs/libgsl.a $(LIBPATH)/gsl-2.8/cblas/.libs/libgslcblas.a -lcurl -lz 

# for binary program
BIN	= glafic
OBJ_BIN	= glafic.o 
OBJS	= call.o ein_tab.o mass.o util.o fits.o \
	  init.o distance.o gsl_zbrent.o gsl_integration.o \
	  source.o extend.o point.o opt_extend.o opt_lens.o \
	  opt_point.o example.o mock.o calcein.o vary.o \
	  gnfw_tab.o commands.o mcmc.o amoeba_opt.o app_ell.o

# for C library
LIB	= libglafic.a
AR	= ar
AFLAGS	= r

# for python interface
PY	= python/glafic/glafic.so
OBJ_PY	= python.o
CFLAGS3	= -Wall -shared -L$(LIBPATH) 
PY_INC  := $(shell python3-config --includes)
PY_LDS  := $(shell python3-config --ldflags --embed)

default: bin

all: bin lib python

bin: $(OBJ_BIN) $(OBJS)
	$(CC) $(CFLAGS) -o $(BIN) $(OBJ_BIN) $(OBJS) $(CFLAGS2) $(LIBS) 

lib: $(OBJS)
	$(AR) $(AFLAGS) $(LIB) $(OBJS) 

python: $(OBJ_PY) $(OBJS)
	$(CC) $(CFLAGS3) -o $(PY) $(OBJ_PY) $(OBJS) $(PY_LDS) $(PY_LIBS)

python.o:python.c glafic.h 
	$(CC) $(CFLAGS) $(PY_INC) -c $< -o $@

%.o:%.c glafic.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f $(BIN) $(OBJ_BIN) $(OBJS) $(LIB) $(PY) $(OBJ_PY) *~ \#* core* 
