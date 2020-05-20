#include "DLInference.h"
#include "Model.h"
#include "Tensor.h"

auto DLInference::ModelBuildUp(){
    Model m(modelGraph);
    m.restore(modelRestore);
    return m;
}

auto DLInference::TensorBuildUp(std::vector<int64_t> shape, int vecFillNr, auto model, std::string node){
   
    int size = std::accumulate(begin(shape), end(shape), 1, std::multiplies<>());
    std::vector<float> toVec(size);

    std::fill(toVec.begin(), toVec.end(), vecFillNr);

    auto dataTensor = new Tensor(model, node);
    dataTensor->set_data(toVec, shape);

    return dataTensor;
}

std::vector<float> DLInference::Generation() {

    Model model = ModelBuildUp();

    auto inputData = TensorBuildUp(inputShape, inputVecNumber, model, inputNode);
    auto eventEnergy = TensorBuildUp(labelShape, energyValue, model, labelNode);
    auto generatedEvent = new Tensor(model, outputNode);

    auto xinput = TensorBuildUp({100,28,28,1}, 5, model, "x_input");

    model.run({xinput,inputData,eventEnergy}, generatedEvent);

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
