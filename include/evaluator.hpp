#ifndef _EVALUATOR_H
#define _EVALUATOR_H

#include "data_loader.hpp"
#include "model.hpp"

class AccEvaluator {

    public:
        AccEvaluator(FasttextModel*);
        float eval(DataLoader*);
    private:
        FasttextModel* model_;
};

#endif//_EVALUATOR_H
