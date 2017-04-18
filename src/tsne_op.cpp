/**
 * TSNE op implementation.
 *
 * This file uses a typical Caffe2 interface to implement the TSNE algorithm
 * that can take in a set of feature vectors, and produce low-dimensional
 * embeddings out of them.
 *
 * The actual tsne implementation is provided in the bhtsne subfolder, from
 * lvdmarteen's original repository.
 */

// Typical Caffe2 headers that you will usually need: context.h defines the
// CPUContext interface, and operator.h defines the Caffe2 operator interface.
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

// The header file of the TSNE implementation.
#include "tsne.h"

// Please consider putting your implementation under the caffe2 namespace.
namespace caffe2 {

/**
 * The TSNE operator. See the operator schema section below on what its input,
 * output and parameters mean.
 * 
 * Note that since the TSNE algorithm only supports CPU, this operator is a
 * derived class of Operator<CPUContext>.
 */
class TSNEOp final : public Operator<CPUContext> {
 public:
  // This is a helper macro that tells the compiler that we are going to
  // use a lot of the Operator<CPUContext> member objects and methods. See
  // caffe2/core/operator.h for more details.
  USE_OPERATOR_FUNCTIONS(CPUContext);

  // The constructor. A Caffe2 operator constructor should always accept two
  // inputs: an OperatorDef protobuf object, and a pointer to a workspace,
  // which hosts all of the blobs.
  TSNEOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        // Now, below here, there are the set of parameters that this operator
        // will be using. These are also helper macros that are defined in
        // operator.h.
        OP_SINGLE_ARG(int, "dims", dims_, 0),
        OP_SINGLE_ARG(float, "perplexity", perplexity_, 50),
        OP_SINGLE_ARG(float, "theta", theta_, 0.5f),
        OP_SINGLE_ARG(int, "random_seed", random_seed_, 0),
        OP_SINGLE_ARG(int, "max_iter", max_iter_, 1000),
        tsne_(new TSNE()) {
    // CAFFE_ENFORCE is the way we check parameter correctness. For example,
    // in TSNE, we should always have a non-negative dimension. If this is not
    // met, an exception is going to be thrown.
    CAFFE_ENFORCE(
        dims_ > 0,
        "You should specify the number of output dimensions.");
  }

  // RunOnDevice() is the function that you should implement for the operator
  // interface.
  bool RunOnDevice() override {
    // If we have two inputs, then we will skip the random initialization step
    // in TSNE and use the existing values in Input(1) as the initial values.
    bool skip_random_init = (InputSize() == 2);
    // Input(0) gives you a const reference to the first input, which should
    // be a TensorCPU object.
    const TensorCPU& X = Input(0);
    CAFFE_ENFORCE(
        X.ndim() == 2,
        "TSNE expects a 2-dimensional tensor as input.");
    CAFFE_ENFORCE(
        X.IsType<double>(),
        "TSNE expects the input to be of data type double.");
    const int N = X.dim32(0);
    const int D = X.dim32(1);
    // If we have 2 inputs, the second input should be of the correct shape.
    // These sanity checks should be self explanatory.
    if (InputSize() == 2) {
      const TensorCPU& init = Input(1);
      CAFFE_ENFORCE(init.ndim() == 2);
      CAFFE_ENFORCE(init.dim32(0) == N);
      CAFFE_ENFORCE(init.dim32(1) == dims_);
      CAFFE_ENFORCE(init.IsType<double>());
    }
    // In any case, we will get the output TensorCPU object, and reshape it to
    // the correct shape.
    TensorCPU* Y = Output(0);
    Y->Resize(N, dims_);

    // After all these, we will simply start running the tsne algorithm.
    tsne_->run(
        // this gets a const point of the input data. Some times a third party
        // library does not distinguish const pointers, which is the case of
        // TSNE. Thus, we will need a const_cast here too. 
        const_cast<double*>(X.template data<double>()),
        N,
        D,
        // mutable_data gives the underlying storage, and also, if necessary,
        // does the actual memory allocation.
        Y->template mutable_data<double>(),
        dims_,
        perplexity_,
        theta_,
        random_seed_,
        skip_random_init,
        max_iter_);

    // If everything works fine, return true which is needed by the function signature.
    return true;
  }

 protected:
  // The actual parameter member variables.
  int dims_;
  float perplexity_;
  float theta_;
  int random_seed_;
  int max_iter_;
  bool is_test_;
  // The TSNE object that the underlying algorithm is implemented with.
  std::unique_ptr<TSNE> tsne_;
};

// This registeres the TSNEOp into Caffe2's operator registry. Essentially,
// it tells Caffe that if it encounters an operator definition named "TSNE",
// TSNEOp is the one that it needs to instantiate.
REGISTER_CPU_OPERATOR(TSNE, TSNEOp);

// Operator schema. This gives a detailed description of what this operator
// is doing, and its intended input, output, and parameters.
OPERATOR_SCHEMA(TSNE)
    // The operator can take either 1 input, or 2 inputs, where the second input
    // is the pre-initialized TSNE embedding to start the algorithm with.
    .NumInputs(1, 2)
    // The operator produces a single output which is the tsne embedding.
    .NumOutputs(1)
    // If there are two inputs, the second input and the first output should
    // always be in-place, meaning that we will write the embedding into the
    // initialization tensor.
    .EnforceInplace({{1, 0}})
    // This is the detailed documentation of the operator.
    .SetDoc(R"DOC(
The TSNE operator implements the Barnes-Hut t-SNE algorithm described in the
paper: http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

Specifically, it takes in a 2-dimensional tensor of shape (N, D), and produces
a 2-dimensional tensor of shape (N, dim) that contains the t-SNE embedding of
the input. For th meaning of the parameters, refer to the original paper.
)DOC")
    .Arg("dim", "(int, required) the output dimension.")
    .Arg("perplexity", "(float, default 50) the perplexity param.")
    .Arg("theta", "(float, default 0.5) the theta param.")
    .Arg("random_seed", "(int, default 0) the random seed if init needed.")
    .Arg("max_iter", "(int, default 1000) the maximum iteration.")
    .Input(0, "X", "The input N*D tensor.")
    .Input(1, "Y", "(optional, in-place) the initialization of the output.")
    .Output(0, "Y", "The output t-SNE embedding.");

// Gradient registration. This is easy in the TSNE case: you should never call
// grdient on a TSNE operator, because it is not expected to be in a forward
// backward pass.
SHOULD_NOT_DO_GRADIENT(TSNE);

} // namespace caffe2
