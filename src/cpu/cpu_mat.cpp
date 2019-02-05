#include "cpu/cpu_mat.hpp"

#include <stdlib.h>
#include <iostream>

FTMat::FTMat() {
}

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

FTMat FTMat::operator * (const FTMat& o)const {
    return FTMat(m_ * o.m_);
}

FTMat FTMat::operator * (const float i) const {
    return FTMat(m_ * i);
}

FTMat FTMat::operator - (const float i) const {
    FTMat mat(m_.rows(), m_.cols());
    for (int i = m_.rows() - 1; i >= 0; i --) {
        for (int j = m_.cols() -1; j >= 0; j --) {
            mat(i, j) = m_(i, j) - i;
        }
    }
    return mat;
}

FTMat operator * (float i, const FTMat& m) {
    return m*i;
}

FTMat FTMat::operator + (const FTMat& o) const {
    return FTMat(m_ + o.m_);
}

FTMat FTMat::operator - (const FTMat& o) const {
    return FTMat(m_ - o.m_);
}

FTMat FTMat::operator = (const FTMat& o) {
    m_ = o.m_;
    return *this;
}
float& FTMat::operator ()(const int i, const int j) {
    return m_(i, j);
}

float FTMat::at(const int i, const int j) const{
    return m_(i, j);
}

FTMat FTMat::dot(const FTMat& mat) const{
    int rows = m_.rows();
    int cols = m_.cols();

    if (rows != mat.num_row() || cols != mat.num_col()) {
        std::cerr << "mat dot failed " << std::endl;
        this->debug();
        mat.debug();
        exit(-1);
    }

    FTMat ret(rows, cols);
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            ret(i, j) = m_(i, j) * mat.at(i, j);
        }
    }
    return ret;
}
FTMat FTMat::transpose() const{
    return FTMat(m_.transpose());
}

FTMat FTMat::row(int i) {
    return FTMat(m_.row(i));
}

int FTMat::num_row() const {
    return m_.rows();
}

int FTMat::num_col() const {
    return m_.cols();
}

void FTMat::random_init() {

    int finout = m_.rows() + m_.cols();
    float a = 1.0 * sqrt(6.0 / std::min(finout, 1000));
    a = std::max(a, 0.01f);

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

void FTMat::one_init() {

    for (int i = m_.rows()-1; i >= 0; i --) {
        for (int j = m_.cols() - 1; j >= 0; j --) {
            m_(i, j) = 1;
        }
    }
}

void FTMat::debug() const{

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

FTMat relu(FTMat& mat) {
    int rows = mat.num_row();
    int cols = mat.num_col();

    FTMat ret(mat);
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            float v = mat(i, j);
            ret(i, j) = v > 0 ? v : 0;
        }
    }
    return ret;
}
