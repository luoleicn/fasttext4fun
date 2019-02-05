
#include "cpu/mat_factory.hpp"

FTMat MatFactory::create(int rows, int cols) {
    FTMat ret(rows, cols);
    ret.random_init();
    return ret;
}

FTMat MatFactory::create(float* vec, int rows, int cols) {
    FTMat ret(vec, rows, cols);
    return ret;
}

FTMat MatFactory::zeros(int rows, int cols) {
    FTMat ret(rows, cols);
    ret.zero_init();
    return ret;
}
