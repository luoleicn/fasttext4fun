#ifndef _MAT_FACTORY_H
#define _MAT_FACTORY_H

#include "matrix.hpp"

class MatFactory {
    public:
        FTMat create(int, int);
        FTMat create(float*, int, int);
        FTMat zeros(int, int);
};

#endif//_MAT_FACTORY_H
