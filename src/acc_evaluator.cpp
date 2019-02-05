#include "evaluator.hpp"
#include <iostream>

AccEvaluator::AccEvaluator(FasttextModel* model):model_(model) {

}

float AccEvaluator::eval(DataLoader* data) {

    int correct = 0;
    int sz = data->size();
    for (int i = 0; i < sz; i ++) {
        DataTypePtr inst_ptr = (*data)[i];
        int pred_label = model_->predict(inst_ptr);
        if (pred_label == inst_ptr->label) {
            correct ++;
        }
    }

    return 1.0f * correct / sz;
}
