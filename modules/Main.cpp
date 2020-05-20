#include "../inference/include/DLInference.h"
#include "../inference/include/Model.h"
#include "../inference/include/Tensor.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <iomanip>
#include <vector>
#include <fstream>
#include <stdio.h>

int main(int argc,char **argv) {
    
    DLInference session; 
    session.SetModelGraph("../cvae.pb");
    session.SetModelRestore("../checkpoint/progress-20-model.ckpt");
    session.SetInputNode("z_input");
    session.SetLabelNode("y_input");
    session.SetOutputNode("decoder/x_decoder_mean_output");
    session.SetInputShape({100,2});
    session.SetLabelShape({100,10});
    session.SetEnergyValue(1);
    session.SetInputVecNumber(5);

    auto result = session.Generation();

    for (auto it = result.begin() ; it != result.end(); ++it) 
        std::cout  << *it <<" , "<< std::endl;

    // Stream Event to File

    std::ofstream outFile("eventCVAE.txt");
       for (const auto e : result) outFile << e << "\n";
    return 0;

};

