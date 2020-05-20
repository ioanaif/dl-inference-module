#include "DLInference.h"
#include "Model.h"
#include "Tensor.h"


std::vector<float> DLInference::generation() {

    inputSize = std::accumulate(begin(inputShape), end(inputShape), 1, std::multiplies<>());
    labelSize = std::accumulate(begin(labelShape), end(labelShape), 1, std::multiplies<>());
    
    // Create  Model
    Model m(modelGraph);
    m.restore(modelRestore);

    // Create Necesary Tensors

    auto inputData = new Tensor(m, inputNode);
    auto eventEnergy = new Tensor(m, labelNode);
    auto generatedEvent = new Tensor(m, outputNode);
    //auto xinput  = new Tensor(m,xiNode);

    // Feed Data to Tensors

    std::vector<float> inputVec(inputSize);
    std::vector<float> energies(labelSize);
    std::fill(inputVec.begin(), inputVec.end(), inputVecNumber);
    std::fill(energies.begin(), energies.end(), energyValue);


    inputData->set_data(inputVec, inputShape);
    eventEnergy->set_data(energies, labelShape);

    auto xinput  = new Tensor(m,"x_input");
    std::vector<float> xi(78400);
    std::fill(xi.begin(), xi.end(), 5);
    xinput->set_data(xi,{100,28,28,1});
    m.run({xinput,inputData,eventEnergy}, generatedEvent);

    // Get Generated Event Tensor
    std::vector<float> result = generatedEvent->get_data<float>();

    return result;
}

void DLInference::SetModelGraph(const std::string &aModelGraph){
    modelGraph = aModelGraph;
}

void DLInference::SetModelRestore(const std::string &aModelRestore){
    modelRestore = aModelRestore;
}

void DLInference::SetInputNode(const std::string &anInputNode){
    inputNode = anInputNode;
}

void DLInference::SetLabelNode(const std::string &aLabelNode){
    labelNode = aLabelNode;
}

void DLInference::SetOutputNode(const std::string &anOutputNode){
    outputNode = anOutputNode;
}

// {100,2}
void DLInference::SetInputShape(const std::vector<int64_t> &anInputShape){
    inputShape = anInputShape;
}

// labelShape = {100,10};
void DLInference::SetLabelShape(const std::vector<int64_t> &aLabelShape){
    labelShape = aLabelShape;
}

void DLInference::SetEnergyValue(const int &anEnergyValue){
    energyValue = anEnergyValue;
}

void DLInference::SetInputVecNumber(const int &anInputVecNumber){
    inputVecNumber = anInputVecNumber;
}
