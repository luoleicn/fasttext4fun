#include <fstream>
#include <string>
#include <iostream>
#include <list>

#include "model.hpp"

FasttextModel::FasttextModel(const char* feature_file, 
        MatFactory* factory) {

    factory_ = factory;
    std::ifstream fin(feature_file);

    root_ = createNode();

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
        std::string label_name("tree_node_");
        int parent, left, right;

        fin >> parent;
        fin >> left;
        fin >> right;

        if (i == 0) {
            root_->self_id = parent;
        }
        LabelTreeNodePtr parentNode = find_by_id(root_, parent);
        if (parentNode == NULL) {
            std::cerr << "find parentid " << parent << "failed" << std::endl;
            exit(-1);
        }
        LabelTreeNodePtr leftNode = createNode();
        leftNode->self_id = left;
        leftNode->parent = parentNode;
        leftNode->is_left = true;

        LabelTreeNodePtr rightNode = createNode();
        rightNode->self_id = right;
        rightNode->parent = parentNode;
        rightNode->is_left = false;

        parentNode->left = leftNode;
        parentNode->right = rightNode;
    }

    fin.close();
}

FasttextModel::~FasttextModel() {
}

void FasttextModel::init(int num_dim, int num_hidden) {

    dim_ = num_dim;
    num_hidden_ = num_hidden;

    embedding_      = factory_->create(num_feature_, dim_);
    hidden_weights_ = factory_->create(dim_, num_hidden_);
    hidden_bias_    = factory_->zeros(1, num_hidden_);

    std::list<LabelTreeNodePtr> list = {root_};
    while (list.size() > 0) {
        LabelTreeNodePtr front = list.front();
        LabelTreeNodePtr left  = front->left;
        LabelTreeNodePtr right = front->right;

        front->theta = factory_->create(num_hidden_, 1);
        front->bias  = factory_->zeros(1, 1);
        if (left != NULL) {
            list.push_back(left);
        }
        if (right != NULL) {
            list.push_back(right);
        }
        list.pop_front();
    }
}

void FasttextModel::load(const char* fn) {

    std::fstream fin(fn);
    fin >> dim_;
    fin >> num_hidden_;

    //load embedding
    float* vec = load_matrix(fin, num_feature_, dim_);
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

    std::list<LabelTreeNodePtr> list = {root_};
    while (list.size() > 0) {
        LabelTreeNodePtr front = list.front();
        LabelTreeNodePtr left  = front->left;
        LabelTreeNodePtr right = front->right;

        int labelid;
        fin >> labelid;

        LabelTreeNodePtr node = get_node(labelid);
        vec = load_matrix(fin, num_hidden_, 1);
        node->theta = factory_->create(vec, num_hidden_, 1);
        delete vec;

        vec = load_matrix(fin, 1, 1);
        node->bias = factory_->create(vec, 1, 1);
        delete vec;

        if (left != NULL) {
            list.push_back(left);
        }
        if (right != NULL) {
            list.push_back(right);
        }
        list.pop_front();
    }

    fin.close();
}

void FasttextModel::save(const char* fn) {
    std::ofstream fout;
    fout.open(fn, std::ios::out|std::ios::trunc);

    fout << dim_ << " ";
    fout << num_hidden_ << " ";

    save_matrix(fout, embedding_);
    save_matrix(fout, hidden_weights_);
    save_matrix(fout, hidden_bias_);

    std::list<LabelTreeNodePtr> list = {root_};
    while (list.size() > 0) {
        LabelTreeNodePtr front = list.front();
        LabelTreeNodePtr left  = front->left;
        LabelTreeNodePtr right = front->right;

        fout << front->self_id << " ";
        save_matrix(fout, front->theta);
        save_matrix(fout, front->bias);

        if (left != NULL) {
            list.push_back(left);
        }
        if (right != NULL) {
            list.push_back(right);
        }
        list.pop_front();
    }
    fout.close();
}

int FasttextModel::predict(DataTypePtr inst) {

    //avg embedding
    FTMat avg_embedding = factory_->zeros(1, dim_);
    for (int i = 0; i < inst->feat_size; i ++) {
        int fid = inst->feat_lst[i];
        avg_embedding = avg_embedding + embedding_.row(fid);
    }
    avg_embedding = avg_embedding * (1.0 / inst->feat_size);
    //1*num_hidden
    FTMat hidden_mat = avg_embedding * hidden_weights_ + hidden_bias_;
    hidden_mat = relu(hidden_mat);

    LabelTreeNodePtr cur  = root_;
    int ret = cur->self_id;
    while (cur != NULL) {

        float p = prob(hidden_mat, cur);
        ret = cur->self_id;

        if (p >= 0.5 && cur->left != NULL && cur->left->self_id >=0) {
            cur = cur->left;
        }
        else if (p < 0.5 && cur->right != NULL && cur->right->self_id >= 0) {
            cur = cur->right;
        }
        else if (cur->left != NULL && cur->left->self_id >= 0) {
            cur = cur->left;
        }
        else {
            cur = cur->right;
        }
    }
    return ret;
}

float FasttextModel::prob(DataTypePtr inst, LabelTreeNodePtr label) {

    //avg embedding
    FTMat avg_embedding = factory_->zeros(1, dim_);
    for (int i = 0; i < inst->feat_size; i ++) {
        int fid = inst->feat_lst[i];
        avg_embedding = avg_embedding + embedding_.row(fid);
    }
    avg_embedding = avg_embedding * (1.0 / inst->feat_size);

    //1*num_hidden
    FTMat hidden_mat = avg_embedding * hidden_weights_ + hidden_bias_;
    hidden_mat = relu(hidden_mat);

    FTMat outMat = hidden_mat * label->theta + label->bias;

    return 1.0 / (1.0 + exp(0.0 - outMat(0, 0)));
}

float FasttextModel::prob(const FTMat& hidden_mat, LabelTreeNodePtr label) {

    FTMat outMat = hidden_mat * label->theta + label->bias;

    return 1.0 / (1.0 + exp(0.0 - outMat(0, 0)));
}

FTMat FasttextModel::hidden_layer(DataTypePtr inst) {

    //avg embedding
    FTMat avg_embedding = factory_->zeros(1, dim_);
    for (int i = 0; i < inst->feat_size; i ++) {
        int fid = inst->feat_lst[i];
        avg_embedding = avg_embedding + embedding_.row(fid);
    }
    avg_embedding = avg_embedding * (1.0 / inst->feat_size);

    //1*num_hidden
    FTMat hidden_mat = avg_embedding * hidden_weights_ + hidden_bias_;
    hidden_mat = relu(hidden_mat);

    return hidden_mat;
}

LabelTreeNodePtr FasttextModel::get_node(int label) {
    return find_by_id(root_, label);
}


void FasttextModel::save_matrix(std::ofstream& fout, const FTMat& mat) {
    int row = mat.num_row();
    int col = mat.num_col();
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            fout << mat.at(i, j) << " ";
        }
    }
}

float* FasttextModel::load_matrix(std::fstream& fin, int row, int col) {
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

void FasttextModel::debug() {

    std::cout << "debug embedding" << std::endl;
    embedding_.debug();

    std::cout << std::endl << std::endl;
    std::cout << "debug hidden_weights_" << std::endl;
    hidden_weights_.debug();

    std::cout << std::endl << std::endl;
    std::cout << "debug hidden_bias_" << std::endl;
    hidden_bias_.debug();

    std::cout << std::endl << std::endl;
    std::cout << "debug tree" << std::endl;
    std::list<LabelTreeNodePtr> list = {root_};
    while (list.size() > 0) {
        LabelTreeNodePtr front = list.front();
        LabelTreeNodePtr left  = front->left;
        LabelTreeNodePtr right = front->right;

        std::cout << front->self_id << " (";
        std::cout << front->theta.num_row() << "," 
            << front->theta.num_col() << ")  ";
        if (left != NULL) {
            std::cout << "left " << left->self_id << " (";
            std::cout << left->theta.num_row() << "," 
                << left->theta.num_col() << ")  ";
            list.push_back(left);
        }
        if (right != NULL) {
            std::cout << "right " << right->self_id << " (";
            std::cout << right->theta.num_row() << "," 
                << right->theta.num_col() << ")  ";
            list.push_back(right);
        }
        front->theta.debug();
        front->bias.debug();
        std::cout << std::endl;
        list.pop_front();
    }
}

LabelTreeNodePtr FasttextModel::createNode() {
    LabelTreeNodePtr ret;

    ret = new LabelTreeNode;
    ret->self_id = -1;
    ret->is_left = true;
    ret->parent = NULL;
    ret->left = NULL;
    ret->right = NULL;
    return ret;
}

FTMat FasttextModel::get_emb(int fid) {
    return embedding_.row(fid);
}

FTMat FasttextModel::get_hw() {
    return hidden_weights_;
}
FTMat FasttextModel::get_hb() {
    return hidden_bias_;
}
void FasttextModel::add_delta_emb(FTMat& mat, int fid) {

    if (mat.num_row() != 1 || mat.num_col() != dim_) {
        std::cerr << "add_delta_emb failed " << std::endl;
        mat.debug();
        exit(-1);
    }
    for (int i = 0; i < dim_; i ++) {
        embedding_(fid, i) += mat(0, i);
    }
}

void FasttextModel::add_delta_hw(FTMat& mat) {

    if (mat.num_row() != dim_ || mat.num_col() != num_hidden_) {
        std::cerr << "add_delta_hw failed " << std::endl;
        mat.debug();
        exit(-1);
    }
    hidden_weights_ = hidden_weights_ + mat;
}
void FasttextModel::add_delta_hb(FTMat& mat) {

    if (mat.num_row() != 1 || mat.num_col() != num_hidden_) {
        std::cerr << "add_delta_hb failed " << std::endl;
        mat.debug();
        exit(-1);
    }
    hidden_bias_ = hidden_bias_ + mat;
}
