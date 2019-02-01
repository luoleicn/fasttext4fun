#include <iostream>

#include "data_loader.hpp"
#include "model.hpp"
#include "optimizer.hpp"
#include "mat_factory.hpp"
#include "evaluator.hpp"

using namespace std;
 
int main(int args, char** argv)
{
    if (args < 4) {
        cerr << "./trainer feaures train_data val_data" << endl;
        exit(-1);
    }

    const char* feature_file = argv[1];
    const char* train_file   = argv[2];
    const char* val_file     = argv[3];

    //create model
    MatFactory* factory  = new CpuMatFactory();
    FasttextModel* model = new FasttextModel(feature_file, factory);
    model->init();

    //create data reader
    DataLoader* train_data = new DataLoader(corpus_file);
    DataLoader* val_data = new DataLoader(corpus_file);

    //create optimizer
    Optimizer* opt = new SGDOptimizer(1e-4);

    //create evaluator
    AccEvaluator* eval = new AccEvaluator(model);

    //train model
    for (int epoch = 0; epoch < 1000; epoch ++ ) {

        data->shuffle();
        for (int i = data.size() - 1; i >= 0; i --) {
            opt->step(model, data[i]);
        }
        float train_acc = eval->eval(train_data);
        float val_acc   = eval->eval(val_data);
        cout << "epoch " << epoch << " train_acc  = " 
            << train_acc << " val_acc = " << val_acc << endl;
    }
    model->save();

    //cleanup
    delete factory;
    delete model;
    delete train_data;
    delete val_data;
    delete eval;

    return 0;
}
