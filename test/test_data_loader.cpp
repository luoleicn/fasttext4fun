#include <gtest/gtest.h>
#include <iostream>

#include "data_loader.hpp"

TEST(Test_dataloader, show) {

    using namespace std;
     
    DataLoader data("../tools/test.dat");
    cout << "data size " << data.size() << endl;

    DataTypePtr ptr = data[0];
    cout << "label " << ptr->label << endl;
    int fsize = ptr->feat_size;
    for (int i = 0; i < fsize; i ++)
        cout << ptr->feat_lst[i] << " ";
    cout << endl;

    ptr = data[data.size()-1];
    cout << "label " << ptr->label << endl;
    fsize = ptr->feat_size;
    for (int i = 0; i < fsize; i ++)
        cout << ptr->feat_lst[i] << " ";
    cout << endl;

    data.shuffle();

    ptr = data[data.size()-1];
    cout << "label " << ptr->label << endl;
    fsize = ptr->feat_size;
    for (int i = 0; i < fsize; i ++)
        cout << ptr->feat_lst[i] << " ";
    cout << endl;
}

