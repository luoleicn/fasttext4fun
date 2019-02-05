#ifndef _SGD_OPTIMIZER_H
#define _SGD_OPTIMIZER_H

#include "optimizer.hpp"

class SGDOptimizer : public Optimizer {
    public:
        SGDOptimizer(float lr, float decay=0);
        void step(FasttextModel*, DataTypePtr);
    private:
        inline float L_p(float p, int y);
        inline float p_o(float o);
        inline float o_bo();
        inline FTMat o_wo(const FTMat& a);
        inline FTMat o_a(const FTMat& w);
        inline FTMat a_h(FTMat& h);
        inline FTMat h_H(const FTMat& avg_embedding);
        inline FTMat h_bh(int num_hidden);
        inline FTMat h_eavg(const FTMat& H);
        inline FTMat eavg_e(int emb_dim, int feat_size);
    private:
        float learning_rate_;
        float decay_; 
};

inline FTMat SGDOptimizer::eavg_e(int emb_dim, int feat_size) {

    float tmp = 1.0f / feat_size;
    FTMat mat(1, emb_dim);
    for (int i = 0; i < emb_dim; i ++) {
        mat(0, i) = tmp;
    }
    return mat;
}

inline FTMat SGDOptimizer::h_eavg(const FTMat& H) {
    return H;
}

inline FTMat SGDOptimizer::h_bh(int num_hidden) {
    FTMat mat(1, num_hidden);
    for (int i = 0; i < num_hidden; i ++)
        mat(0, i) = 1;
    return mat;
}

inline FTMat SGDOptimizer::h_H(const FTMat& avg_embedding) {
    return avg_embedding;
}

inline FTMat SGDOptimizer::a_h(FTMat& h) {
    int rows = h.num_row();
    int cols = h.num_col();

    FTMat ret(rows, cols);
    for (int i = 0; i < rows; i ++) {
        for (int j = 0; j < cols; j ++) {
            ret(i, j) = h(i, j) > 0 ? 1 : 0;
        }
    }
    return ret;
}
inline FTMat SGDOptimizer::o_a(const FTMat& w) {
    return w.transpose();
}

inline FTMat SGDOptimizer::o_wo(const FTMat& a) {
    return a.transpose();
}

inline float SGDOptimizer::o_bo() {
    return 1.0f;
}

inline float SGDOptimizer::p_o(float p) {
    return p * (1-p);
}

inline float SGDOptimizer::L_p(float p, int y) {
    return -1.0f * y / (p+1e-4) + (1-y) / (1.0f-p+1e-4);
}
#endif//_SGD_OPTIMIZER_H
