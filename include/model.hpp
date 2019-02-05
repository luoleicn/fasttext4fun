#ifndef _MODEL_H
#define _MODEL_H

#include "data_loader.hpp"
#include "matrix.hpp"

//层次化label
typedef struct LabelTreeNode {

    int self_id;
    struct LabelTreeNode *parent;
    struct LabelTreeNode *left, *right;
    //当前节点是左孩子还是右孩子
    bool is_left;
    //num_hidden * 1
    FTMat theta;
    //1*1
    FTMat bias;

}LabelTreeNode, *LabelTreeNodePtr;

/*
 * 参数更新：
 * http://note.youdao.com/noteshare?id=9de1cacabb533b7fc61723303c4bfb53
 */
class FasttextModel {

    public:
        friend class SGDOptimizer;//for debug only

        FasttextModel(const char* feature_file, MatFactory*);
        ~FasttextModel();
        void init(int num_dim, int num_hidden);
        void load(const char* fn="model.bin");
        void save(const char* model_file="model.bin");
        int predict(DataTypePtr);
        float prob(DataTypePtr inst, LabelTreeNodePtr label);
        float prob(const FTMat& hidden_mat, LabelTreeNodePtr label);
        FTMat hidden_layer(DataTypePtr);
        LabelTreeNodePtr get_node(int label);
    public:
        FTMat get_emb(int);
        FTMat get_hw();
        FTMat get_hb();
        void add_delta_emb(FTMat&, int);
        void add_delta_hw(FTMat&);
        void add_delta_hb(FTMat&);
    public:
        inline int get_emb_dim() {return dim_;}
        inline int get_hidden_dim() {return num_hidden_;}
        void debug();
    private:
        LabelTreeNodePtr find_by_id(LabelTreeNodePtr, int);
        float* load_matrix(std::fstream&, int row, int col);
        void save_matrix(std::ofstream& fout, const FTMat& mat);
        LabelTreeNodePtr createNode();
    private:
        int num_feature_;
        int num_label_;
        int num_hidden_;
        int dim_;

        LabelTreeNodePtr root_;
        MatFactory* factory_;

        //num_feat * dim_
        FTMat embedding_;
        //dim_ * num_hidden_
        FTMat hidden_weights_;
        //1 * num_hidden_
        FTMat hidden_bias_;
};

#endif//_MODEL_H
