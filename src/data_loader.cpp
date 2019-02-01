#include "data_loader.hpp"

#include <iostream>
#include <fstream>
#include <string>

DataLoader::DataLoader(const char* fn):file_name_(fn) {

    inst_num_ = 0;

    std::ifstream fin(fn);
    std::string line;
    while (fin.getline(line)) {
        inst_num_ ++;
    }
    fin.reset();

    data_lst_ = new DataType[inst_num_];

    int i = 0;
    while (fin.getline(line)) {
        parseLine(line, &data_lst_[i]);
        i ++;
    }
    fin.close();
}

DataLoader::~DataLoader() {
    delete [] data_lst_;
}

size_t DataLoader::size() {
    return inst_num_;
}

void DataLoader::shuffle() {
    std::cerr << "shuffle is not implemented" << endl;
    exit(-1);
}

DataTypePtr DataLoader::operator [](int i) {
    return &data_lst_[i];
}

void DataLoader::parseLine(std::string& line, DataTypePtr ret) { 
    cerr << "parse line is not implemented" << endl;
    exit(-1);
}


