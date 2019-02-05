#include <gtest/gtest.h>
#include <iostream>

#include "matrix.hpp"

TEST(Test_cpu_mat, show) {

    using namespace std;
     
    FTMat mat1(3, 2);
    cout << "mat1 : " << endl;
    mat1.debug();

    float a[6];
    for (int i = 0; i < 6; i ++) 
        a[i] = i;

    FTMat mat2(a, 3, 2);
    cout << "mat2 : " << endl;
    mat2.debug();

    FTMat mat3(mat2);
    cout << "mat3 : " << endl;
    mat3.debug();

    FTMat mat4 = mat2;
    mat2.zero_init();
    cout << "mat2 : " << endl;
    mat2.debug();

    cout << "mat4 : " << endl;
    mat4.debug();

    FTMat mat5 = mat3 + mat4;
    cout << "mat5 : " << endl;
    mat5.debug();

    cout << "mat3.transpose()" << endl;
    mat3.transpose().debug();

    FTMat mat6 = mat3.transpose() * mat4;
    cout << "mat6 :" << endl;
    mat6.debug();

    cout << "mat6(1, 1) :" << endl;
    cout << mat6(1, 1) << endl;

    cout << "mat6 row(1) :" << endl;
    mat6.row(1).debug();

    cout << "mat2 random_init :" << endl;
    mat2.random_init();
    mat2.debug();

    cout << "mat2 :" << endl;
    mat2(0, 0) = 10;
    mat2.debug();

    FTMat mat7 = relu(mat2);
    cout << "mat7 :" << endl;
    mat7.debug();
}

