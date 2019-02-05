
#include "optimizer/sgd_optimizer.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

SGDOptimizer::SGDOptimizer(float lr, float decay)
    :learning_rate_(lr), decay_(decay){}

/*
 * 参数更新和代码里的符号意义
 * http://note.youdao.com/noteshare?id=9de1cacabb533b7fc61723303c4bfb53
 */
void SGDOptimizer::step(FasttextModel* model, DataTypePtr data) {

    
    int emb_dim = model->get_emb_dim();
    int hid_dim = model->get_hidden_dim();

    int ans_label = data->label;
    LabelTreeNodePtr node = model->get_node(ans_label);
    LabelTreeNodePtr father = node->parent;

    FTMat hH_grad(emb_dim, hid_dim);
    hH_grad.zero_init();
    
    FTMat hb_grad(1, hid_dim);
    hb_grad.zero_init();

    FTMat avg_embedding = FTMat(1, emb_dim);
    avg_embedding.zero_init();
    FTMat* embedding_grad = new FTMat[data->feat_size];
    for (int i = 0; i < data->feat_size; i ++) {
	int fid = data->feat_lst[i];
	avg_embedding = avg_embedding + model->get_emb(fid);

        FTMat zeros(1, emb_dim);
        zeros.zero_init();
        embedding_grad[i] = zeros;
    }
    avg_embedding = avg_embedding * (1.0 / data->feat_size);
    FTMat hidden_layer = model->hidden_layer(data);

    int count = 0;
    while (father != NULL) {
        count ++;

        float p = model->prob(hidden_layer, father);
        int   y = node->is_left ? 1 : 0;
        FTMat a = relu(hidden_layer);

        float d_L_p  = L_p(p, y);
        float d_p_o  = p_o(p);
        float d_o_bo = o_bo();
        FTMat d_o_wo = o_wo(a);
        FTMat d_o_a  = o_a(father->theta);
        FTMat d_a_h  = a_h(hidden_layer);
        FTMat d_h_H  = h_H(avg_embedding);
        FTMat d_h_bh = h_bh(hid_dim);
        FTMat d_h_eavg = h_eavg(model->get_hw());
        FTMat d_eavg_e = eavg_e(emb_dim, data->feat_size);

        if (false) {
            using namespace std;
            cout << "p " << p << endl;
            cout << "y " << y << endl;
            cout << "L_p : " << d_L_p << endl;
            cout << "p_o : " << d_p_o << endl;
            cout << "o_a : " << endl;
            d_o_a.debug();
            cout << "o_wo : " << endl;
            d_o_wo.debug();
            cout << "a_h : " << endl;
            d_a_h.debug();
            cout << "h_eavg : " << endl;
            d_h_eavg.debug();
            cout << "evag_e : " << endl;
            d_eavg_e.debug();
            cout << "h_bh : " << endl;
            d_h_bh.debug();
        }

        for (int i = 0; i < data->feat_size; i ++) {
            embedding_grad[i] = (1.0-decay_) * embedding_grad[i]
                            - learning_rate_
                            * d_L_p
                            * d_p_o
                            * (d_o_a
                            .dot(d_a_h)
                            * d_h_eavg.transpose())
                            .dot(d_eavg_e);
        }

        hb_grad = (1.0-decay_) * hb_grad 
                        - learning_rate_
                        * d_L_p
                        * d_p_o
                        * d_o_a
                        .dot(d_a_h)
                        .dot(d_h_bh);

        hH_grad = (1.0-decay_) * hH_grad 
                        - learning_rate_
                        * d_L_p
                        * d_p_o
                        * d_h_H.transpose()
                        * (d_o_a
                        .dot(d_a_h));
        
        father->theta = (1.0-decay_) * father->theta 
                                - (learning_rate_ 
                                * d_L_p
                                * d_p_o
                                * d_o_wo);
        
        father->bias = (1.0-decay_) * father->bias 
                                - learning_rate_ 
                                * d_L_p
                                * d_p_o
                                * d_o_bo;
        
        node = father;
        father = father->parent;
    }

    float mean = 1.0f / count;

    hH_grad = hH_grad * mean;
    model->add_delta_hw(hH_grad);

    hb_grad = hb_grad * mean;
    model->add_delta_hb(hb_grad);

    for (int i = 0; i < data->feat_size; i ++) {
        embedding_grad[i] = embedding_grad[i] * mean;
        model->add_delta_emb(embedding_grad[i], data->feat_lst[i]);
    }
    delete [] embedding_grad;
}

