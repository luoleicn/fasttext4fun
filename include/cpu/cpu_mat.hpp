#ifndef _CPU_MAT_H
#define _CPU_MAT_H

#include <Eigen/Dense>

class FTMat{
    public:
        FTMat(int row, int col);
        FTMat(float*, int row, int col);
        FTMat(const FTMat&);

        FTMat operator * (const FTMat&);
        FTMat operator + (const FTMat&);
        FTMat operator = (const FTMat&);
        float operator ()(const int, const int);
        FTMat transpose();
        FTMat row(int);
        int num_row();
        int num_col();
        void random_init();
        void zero_init();
        void debug();
    private:
        FTMat(const Eigen::MatrixXf&);
    private:
        Eigen::MatrixXf m_;
};

#endif//_CPU_MAT_H
