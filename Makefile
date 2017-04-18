# The example makefile that you will need to build your custom Caffe2
# extension.

# CAFFE2_INCLUDE and CAFFE2_LIB should point to your local Caffe2
# path. If you have done cmake install for Caffe2 and it lives under
# your default system folder, you may not need these two.
CAFFE2_INCLUDE = /Users/jiayq/Research/caffe2/build/install/include
CAFFE2_LIB = /Users/jiayq/Research/caffe2/build/install/lib/

# Depending on your Caffe2 build, you may need to add different dependencies
# to the link flag. In our case, Caffe2 is built with glog, gflags and
# protobuf, and since we have used these functionalities in our code, we will
# link to these dependencies as well.
CAFFE2_LINKFLAGS = "-L$(CAFFE2_LIB) -lCaffe2_CPU -lglog -lgflags -lprotobuf"

TARGET = caffe2_tsne
SOURCE_FILES = src/tsne_op.cpp bhtsne/sptree.cpp bhtsne/tsne.cpp


STATIC_LIB := lib$(TARGET).a
SHARED_LIB := lib$(TARGET).so

CXXFLAGS := -std=c++11 -I$(CAFFE2_INCLUDE) -Ibhtsne -I/usr/local/include/eigen3

OBJECTS = $(SOURCE_FILES:.cpp=.o)

all: $(STATIC_LIB) $(SHARED_LIB)

clean:
	rm $(SHARED_LIB)
	rm $(STATIC_LIB)
	rm $(OBJECTS)

$(SHARED_LIB): $(OBJECTS)
	$(CXX) -shared -o $(SHARED_LIB) $(OBJECTS) 

$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $(OBJECTS)

%.o : %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
