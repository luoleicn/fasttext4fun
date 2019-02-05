#ifndef _CPU_MAT_H
#define _CPU_MAT_H

#include <Eigen/Dense>

class FTMat{
    public:
        FTMat();
        FTMat(int row, int col);
        FTMat(float*, int row, int col);
        FTMat(const FTMat&);

        FTMat operator * (const FTMat&) const;
        FTMat operator * (const float) const;
        FTMat operator + (const FTMat&) const;
        FTMat operator - (const float) const;
        FTMat operator - (const FTMat&) const;
        FTMat operator = (const FTMat&);
        float& operator ()(const int, const int);
        FTMat dot(const FTMat&)const;

        float at(const int, const int) const;
        FTMat transpose()const;
        FTMat row(int);
        int num_row()const;
        int num_col()const;
        void random_init();
        void zero_init();
        void one_init();
        void debug()const;
    private:
        FTMat(const Eigen::MatrixXf&);
    private:
        Eigen::MatrixXf m_;
};

FTMat operator * (float, const FTMat&);
FTMat relu(FTMat&);

#endif//_CPU_MAT_H
