#ifndef _MATRIX_H
#define _MATRIX_H

class FTMat {
    public:
        virtual FTMat operator * (FTMat&) = 0;
        virtual FTMat operator + (FTMat&) = 0;
        virtual FTMat transpose() = 0;
        virtual FTMat row() = 0;
};
#endif//_MATRIX_H
