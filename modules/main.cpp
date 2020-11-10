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
#include <iostream>
#include <string>
#include "HelperFun.h"

using namespace std;

vector<int64_t> ShapeTransform(string s) {istringstream is(s); vector<int64_t> shape ( ( std::istream_iterator<int>( is ) ), ( std::istream_iterator<int>() ) ); return shape;}


int main(int argc,char **argv) {
    
    DLInference session; 
    HelperFun helps;

    vector<string> modelVars; 
    string line;
    string path = "../AR_latest_params.txt";
    ifstream file;
    file.open(path,ios::in);

    if(!file){
        cout<<"Error in opening file "<<path<<'\n';
        return 0;
    }

    while(getline(file, line)) {
	modelVars.push_back(line);
    }

    file.close();

    session.SetModelGraph(modelVars[0]);
    session.SetModelRestore(modelVars[1]);
    session.SetInputNode(modelVars[2]);
    session.SetLabelNode(modelVars[3]);
    session.SetOutputNode(modelVars[4]);
    session.SetInputShape(ShapeTransform(modelVars[5]));
    session.SetLabelShape(ShapeTransform(modelVars[6]));
    auto energyValue = helps.LabelTransform(12969);
    session.SetEnergyValue(energyValue);
    session.SetInputVecNumber(0.7);


//    session.SetModelGraph("../checkpoint/graphPxRW.pb");
//    session.SetModelRestore("../checkpoint/params_PbWO4.ckpt");
//    session.SetInputNode("model/input_cells");
//    session.SetLabelNode("model/input_labels");
//    session.SetOutputNode("mul");
//    session.SetInputShape({12,8,8,24});
//    session.SetLabelShape({12});
//    float energy = 5.5;
//    float input_vec = 6.5;
//    session.SetEnergyValue(energy);
//    session.SetInputVecNumber(input_vec);

//     session.SetModelGraph("../cvae.pb");
//     session.SetModelRestore("../checkpoint/progress-20-model.ckpt");
//     session.SetInputNode("z_input");
//     session.SetExtraInputNode("x_input");
//     session.SetLabelNode("y_input");
//     session.SetOutputNode("decoder/x_decoder_mean_output");
//     session.SetInputShape({100,2});
//     session.SetExtraInputShape({100,28,28,1});
//     session.SetLabelShape({100,10});
//     session.SetExtraInputVecNumber(5);

    //GAN DETAILS 

//   session.SetModelGraph("../dcgan.pb");
//   session.SetModelRestore("../checkpoint/model.b64.ckpt");
//   session.SetInputNode("z_input");
//   session.SetLabelNode("y_input");
//   session.SetOutputNode("generator/gen_output");
//   session.SetInputShape({64,100});
//   session.SetLabelShape({64,10});
//   std::cout  << session.GetModelGraph() << std::endl;
//   session.SetEnergyValue(1);
//   session.SetInputVecNumber(5);


    auto result = session.Generation();

    vector<float> energies;
    for (int i = 0; i < 13824; i++) {
	energies.push_back(helps.DepositionsTransform(result[i]));
        cout<<energies[i]<<'\n'; 
   }
    ;
    cout<<"SIZE OF RESULT: "<<energies.size()<<std::endl;

    auto pos = helps.CellPositionsGeneration();
    cout<<"SIZE OF POS: "<<pos.size()<<std::endl;
    // Stream Event to File

    std::ofstream outFile("../eventAR2.txt");
       for (const auto e : energies) outFile << e << "\n";

    return 0;

};


