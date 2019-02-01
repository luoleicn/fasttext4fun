#ifndef _MODEL_H
#define _MODEL_H

#include "data_loader.hpp"
#include "mat_factory.hpp"

//层次化label
typedef struct LabelTreeNode {

    int self_id;
    struct LabelTreeNode *parent;
    struct LabelTreeNode *left, *right;

}LabelTreeNode, *LabelTreeNodePtr;

class FasttextModel {

    public:
        FasttextModel(const char* feature_file, MatFactory*);
        ~FasttextModel();
        void init(int num_dim, int num_hidden);
        void load(const char* fn);
        int predict(DataTypePtr);
    private:
        LabelTreeNodePtr find_by_id(int);
        float* load_matrix(std::fstream&, int row, int col);
    private:
        int num_feature_;
        int num_label_;
        int num_hidden_;
        int dim_;

        LabelTreeNodePtr root_;
        MatFactory* factory_;

        //num_feat * dim_
        FTMat* embedding_;
        //dim_ * num_hidden_
        FTMat* hidden_weights_;
        //1 * num_hidden_
        FTMat* hidden_bias_;
        //num_label_ * num_hidden_
        FTMat* output_weights_;
        //num_label_ * 1
        FTMat* output_bias_;
}

#endif//_MODEL_H
