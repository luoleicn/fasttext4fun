#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include "model.hpp"

class Optimizer {
    public:
        virtual void step(FasttextModel*, DataTypePtr)=0;
};
#endif//_OPTIMIZER_H
