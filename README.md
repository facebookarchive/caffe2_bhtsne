# caffe2_bhtsne

## What's this?

[Caffe2](https://github.com/caffe2/caffe2) is an open-source deep learning library that is designed to be modular in nature. For many deep learning libraries, especially those earlier ones such as [Caffe](https://github.com/BVLC/caffe), one either only sticks with only the operators provided by the framework, or has to do non-trivial modifications to the codebase in order to compile custom implementation into the library.

This repository gives an example on how you can develop your custom operators and functionalities independent from the main Caffe2 codebase, and "plug-in" these into Caffe2 by dynamically loading your operators. The benefit is that

- You can maintain your own library at your own pace.
- You have a better control over the codebase.
- It becomes easier to share code instead of having to merge and constantly upstream changes from the main Caffe2 library.

## How should I use this?

Imagine you are developing a new deep learning algorithm, and in your method you have some new layers / operators that existing frameworks do not cover. At the same time, you still want to take advantage of the existing framework functionalities. What you can do is to write them as standard Caffe2 operators, build your code as a shared library, and then load it either from C++ or Python.

We will demonstrate the mechanism using a popular software package, T-SNE, and show how one can create a Caffe2 operator that produces T-SNE embeddings from an input Tensor, living under the Caffe2 runtime.

## Let's do this.