#include <iostream>

#include "evaluator.hpp"
#include "data_loader.hpp"

using namespace std;
 
int main(int args, char** argv)
{
    if (args < 4) {
        cerr << "./test model_file test_file feature_file" << endl;
        exit(-1);
    }

    const char* feature_file = argv[3];
    const char* model_file = argv[1];
    const char* test_file  = argv[2];

    //create model
    MatFactory* factory  = new CpuMatFactory();
    FasttextModel* model = new FasttextModel(feature_file, factory);
    model->load(model_file);

    //create evaluator
    Evaluator* eval = new AccEvaluator();

    //create data reader
    DataLoader* test_data = new DataLoader(test_file);

    float acc = eval->eval(test_data);

    cout << "test acc " << acc << endl;

    //cleanup
    delete test_data;
    delete eval;
    delete model;
    delete factory;

    return 0;
}
