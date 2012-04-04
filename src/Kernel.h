#ifndef KERNEL_H
#define KERNEL_H

#include "Vector.h"
#include "Matrix.h"


template <typename TPrecision, typename TKernelParam>
class Kernel{

  public:
    virtual ~Kernel(){};

    virtual TPrecision f(Vector<TPrecision> &x1, Vector<TPrecision> &x2) = 0;
    virtual TPrecision f(Vector<TPrecision> &x1, Matrix<TPrecision> &X2, int i2) = 0;
    virtual TPrecision f(Matrix<TPrecision> &X1, int i1, Matrix<TPrecision> &X2, int i2) = 0;


    virtual void grad(Vector<TPrecision> &x, Vector<TPrecision> &x2, Vector<TPrecision> &g) = 0;
    virtual TPrecision gradf(Vector<TPrecision> &x, Vector<TPrecision> &x2, Vector<TPrecision> &g) = 0;

    virtual TKernelParam gradKernelParam(Vector<TPrecision> &x, Vector<TPrecision> &x2) = 0;
    
    virtual void setKernelParam(TKernelParam param) = 0;

    virtual TKernelParam getKernelParam() = 0;

};

#endif
