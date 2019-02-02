#include "cpu/cpu_mat.hpp"

#include <stdlib.h>
#include <iostream>

FTMat::FTMat(int row, int col) {
    m_ = Eigen::MatrixXf(row, col);
}


FTMat::FTMat(float* vec, int row, int col) {
    m_ = Eigen::MatrixXf(row, col);

    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            m_(i, j) = vec[i*col+j];
        }
    }
}

FTMat::FTMat(const FTMat& o) {
    m_ = o.m_;
}


FTMat::FTMat(const Eigen::MatrixXf& m):m_(m) {
}

FTMat FTMat::operator * (const FTMat& o) {
    return FTMat(m_ * o.m_);
}

FTMat FTMat::operator + (const FTMat& o) {
    return FTMat(m_ + o.m_);
}

FTMat FTMat::operator = (const FTMat& o) {
    return FTMat(o);
}
float FTMat::operator ()(const int i, const int j) {
    return m_(i, j);
}

FTMat FTMat::transpose() {
    return FTMat(m_.transpose());
}

FTMat FTMat::row(int i) {
    return FTMat(m_.row(i));
}

int FTMat::num_row() {
    return m_.rows();
}

int FTMat::num_col() {
    return m_.cols();
}

void FTMat::random_init() {

    float a = 1.0 * sqrt(6.0 / (m_.rows() + m_.cols()));
    for (int i = m_.rows()-1; i >= 0; i --) {
        for (int j = m_.cols() - 1; j >= 0; j --) {
            m_(i, j) = (rand() / double(RAND_MAX) * 2 - 1) * a;
        }
    }
}

void FTMat::zero_init() {

    for (int i = m_.rows()-1; i >= 0; i --) {
        for (int j = m_.cols() - 1; j >= 0; j --) {
            m_(i, j) = 0;
        }
    }
}

void FTMat::debug() {

    int num_row = m_.rows();
    int num_col = m_.cols();

    std::cout << "matrix size " << m_.rows() << " "
        << m_.cols() << std::endl;
    for (int i = 0; i < num_row; i ++) {
        for (int j = 0; j < num_col; j ++) {
            std::cout << m_(i,j) << " ";
        }
        std::cout << std::endl;
    }
    
}
