#ifndef _DATA_LOADER_H
#define _DATA_LOADER_H

#include <string>

typedef struct DataType {
    int label;
    int* feat_lst;
    int feat_size;
}DataType, *DataTypePtr;

class DataLoader {

    public:
        DataLoader(const char*);
        ~DataLoader();
        size_t size();
        void shuffle();
        DataTypePtr operator [](int);
    private:
        void parseLine(std::string&, DataTypePtr);
    private:
        const char* file_name_;
        DataTypePtr data_lst_;
        int inst_num_;
};

#endif//_DATA_LOADER_H
