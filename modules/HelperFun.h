#ifndef HELPERFUN_HelperFun_H
#define HELPERFUN_HelperFun_H

#include <iostream>
#include <vector>  
#include <array>
class HelperFun {

public:

    std::vector<float> LabelTransform(float l);
    float DepositionsTransform(const float i);
    std::vector<std::array<double,3>> CellPositionsGeneration();
};

#endif //HELPERFUN_helperFun_H
