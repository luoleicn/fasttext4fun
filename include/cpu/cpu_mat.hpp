#ifndef _CPU_MAT_H
#define _CPU_MAT_H

#include <Eigen/Dense>

#include "matrix.hpp"

class CPUFTMat : public FTMat {
    public:
        FTMat operator * (FTMat&) = 0;
        FTMat operator + (FTMat&) = 0;
        FTMat transpose() = 0;
        FTMat row() = 0;
    private:
        Eigen::MatrixXf m_;
};

#endif//_CPU_MAT_H
