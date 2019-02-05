#include <gtest/gtest.h>
#include <iostream>

#include "model.hpp"

TEST(Test_model, show) {

    using namespace std;
     
    const char* feature_file = "../tools/featurefile";
    MatFactory* factory  = new MatFactory();
    FasttextModel* model = new FasttextModel(feature_file, factory);
    model->init(3, 5);
    model->debug();
}

