#include "data_loader.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>    
#include <random>
#include <sstream>

DataLoader::DataLoader(const char* fn):file_name_(fn) {

    inst_num_ = 0;

    std::ifstream fin(fn);
    if (!fin.is_open()) {
        std::cerr << "cannot open file " << fn << std::endl;
        exit(-1);
    }
    std::string line;
    while (getline(fin, line)) {
        inst_num_ ++;
    }
    fin.clear();
    fin.seekg(0, std::ios::beg);

    data_lst_ = new DataType[inst_num_];

    int i = 0;
    while (getline(fin, line)) {
        parseLine(line, &data_lst_[i]);
        i ++;
    }
    fin.close();
}

DataLoader::~DataLoader() {
    for (int i = 0; i < inst_num_; i ++)
        delete [] data_lst_[i].feat_lst;
    delete [] data_lst_;
}

size_t DataLoader::size() {
    return inst_num_;
}

void DataLoader::shuffle() {
    std::shuffle(data_lst_, data_lst_+inst_num_,
            std::default_random_engine(0));
}

DataTypePtr DataLoader::operator [](int i) {
    return &data_lst_[i];
}

void DataLoader::parseLine(std::string& line, DataTypePtr ret) { 
    auto tokens = split(line, ' ');
    auto iter   = tokens.begin();

    ret->label = atoi(iter->c_str());
    iter ++;
    ret->feat_size = tokens.size() - 1;
    ret->feat_lst  = new int[ret->feat_size];
    for (int i = 0; i < ret->feat_size; i ++) {
        ret->feat_lst[i] = atoi(iter->c_str());
        iter ++;
    }
}

std::vector<std::string> DataLoader::split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
   }
   return tokens;
}



