#include <fstream>
#include <string>

#include "model.hpp"

FasttextModel::FasttextModel(const char* feature_file, 
        MatFactory* factory) {

    factory_ = factory;
    std::ifstream fin(feature_file);

    root_ = new LabelTreeNode;
    root_->self_id = -1;
    root_->left = NULL;
    root_->right = NULL;

    fin >> num_feature_;
    for (int i = 0; i < num_feature_; i ++) {
        int fid;
        std::string feature_name;

        fin >> fid;
        fin >> feature_name;
    }

    fin >> num_label_;
    for (int i = 0; i < num_label_; i ++) {
        int labelid;
        std::string label_name;

        fin >> labelid;
        fin >> label_name;
    }

    int tree_size;
    fin >> tree_size;
    for (int i = 0; i < tree_size; i ++) {
        num_label_ ++;
        std::string label_name("tree_node_");
        int parent, left, right;

        fin >> parent;
        fin >> left;
        fin >> right;

        LabelTreeNodePtr parentNode = find_by_id(parent, parent);
        if (parentNode == NULL) {
            std::cerr << "find parentid " << parent << "failed" << endl;
            exit(-1);
        }
        LabelTreeNodePtr leftNode = new LabelTreeNode;
        leftNode->self_id = left;
        leftNode->left = NULL;
        leftNode->right = NULL;

        LabelTreeNodePtr rightNode = new LabelTreeNode;
        rightNode->self_id = right;
        rightNode->left = NULL;
        rightNode->right = NULL;

        parentNode->left = leftNode;
        parentNode->right = rightNode;
    }

    fin.close();
}

FasttextModel::~FasttextModel() {
    delete embedding_;
    delete hidden_weights_;
    delete hidden_bias_;
    delete output_weights_;
    delete output_bias_;
}

void FasttextModel::init(int num_dim, int num_hidden) {

    dim_ = num_dim;
    num_hidden_ = num_hidden;

    embedding_      = factory_->create(num_feature_, dim_);
    hidden_weights_ = factory_->create(dim_, num_hidden_);
    hidden_bias_    = factory_->create(1, num_hidden_);
    output_weights_ = factory_->create(num_label_, num_hidden_);
    output_bias_    = factory_->create(num_label_, 1);
}

void FasttextModel::load(const char* fn) {

    std::fstream fin(fn);
    fin >> dim_;
    fin >> num_hidden_;

    //load embedding
    vec = load_matrix(fin, num_feature_, dim_);
    embedding_ = factory_->create(vec, num_feature_, dim_);
    delete vec;

    //load hidden weights
    vec = load_matrix(fin, dim_, num_hidden_);
    hidden_weights_ = factory_->create(vec, dim_, num_hidden_);
    delete vec;

    //load hidden bias
    vec = load_matrix(fin, 1, num_hidden_);
    hidden_bias_ = factory_->create(vec, 1, num_hidden_);
    delete vec;

    //load output weights
    vec = load_matrix(fin, num_hidden_, num_label_);
    output_weights_ = factory_->create(vec, num_hidden_, num_label_);
    delete vec;

    //load output bias
    vec = load_matrix(fin, num_label_, 1);
    output_bias_ = factory_->create(vec, num_label_, 1);
    delete vec;

    fin.close();
}

int predict(DataTypePtr inst) {

    //avg embedding
    FTMat avg_embedding = factory_->zeros(1, dim_);
    for (int i = 0; i < inst->feat_size; i ++) {
        int fid = inst->feat_lst[i];
        avg_embedding = avg_embedding + embedding_->row([fid]);
    }
    avg_embedding = avg_embedding / inst->feat_size;

    int predict_label = -1;
    LabelTreeNodePtr cur  = root_;
    while (cur->leftNode != NULL && cur->rightNode != NULL) {
        LabelTreeNodePtr left = cur->left;
        LabelTreeNodePtr right = cur->right;

        FTMat hidden_mat = avg_embedding * hidden_weights_ 
                                + hidden_bias_;

        FTMat hidden_mat = relu(hidden_mat);
        
        //predict left
        int left_label = left->self_id;
        float lvalue = hidden_mat * output_weights_[left_label].transpose()
                                + output_bias_[left_label].transpose();
        //predict right
        int right_label = right->self_id;
        float rvalue = hidden_mat * output_weights_[right_label].transpose()
                                + output_bias_[right_label].transpose();
        cur = lvalue > rvalue ? left : right;
    }
    return lvalue->self_id;
}

float* FasttextMode::load_matrix(std::fstream& fin, int row, int col) {
    float* vec = new float[row * col];
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            fin >> vec[i*col + j];
        }
    }
    return vec;
}

LabelTreeNodePtr FasttextModel::find_by_id(LabelTreeNodePtr node, int id) {

    if (node == NULL)
        return NULL;

    if (node->self_id == id) 
        return node;

    LabelTreeNodePtr ret = find_by_id(node->left, id);
    if (ret != NULL)
        return ret;
    else
        return find_by_id(node->right, id);
}
