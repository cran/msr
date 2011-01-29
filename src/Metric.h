#ifndef METRIC_H
#define METRIC_H

#include "Matrix.h"
#include "Vector.h"

template<typename TPrecision>
class Metric{
    
  public:
    virtual ~Metric(){};
    virtual TPrecision distance(Vector<TPrecision> &x1, Vector<TPrecision> &x2) = 0;
    virtual TPrecision distance(Matrix<TPrecision> &X, int ix,
        Matrix<TPrecision> &Y, int iy) = 0;
    virtual TPrecision distance(Matrix<TPrecision> &X, int i1, Vector<TPrecision> &x2) = 0;
};


#endif
