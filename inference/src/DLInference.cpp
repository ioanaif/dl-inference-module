#include "DLInference.h"
#include "Model.h"
#include "Tensor.h"

auto DLInference::ModelBuildUp() {
    Model m(modelGraph);
    m.restore(modelRestore);
    return m;
}
auto DLInference::TensorBuildUp(std::vector<int64_t> shape, int vecFillNr, auto &model, std::string node) {
   
    int size = std::accumulate(begin(shape), end(shape), 1, std::multiplies<>());
    std::vector<float> toVec(size);

    std::fill(toVec.begin(), toVec.end(), vecFillNr);

    Tensor dataTensor{model, node};
    dataTensor.set_data(toVec, shape);

    return dataTensor;
}

std::vector<float> DLInference::Generation() {

    Model model = ModelBuildUp();

    auto inputData = TensorBuildUp(inputShape, inputVecNumber, model, inputNode);
    auto eventEnergy = TensorBuildUp(labelShape, energyValue, model, labelNode);
    auto generatedEvent = new Tensor(model, outputNode);

    if (extraInputNode != "") {
        auto xInput = TensorBuildUp(extraInputShape, extraInputVecNumber, model, extraInputNode);
        model.run({&xInput,&inputData,&eventEnergy}, generatedEvent);
    } else {
        model.run({&inputData,&eventEnergy}, generatedEvent);
    }

    // Get Generated Event Tensor
    std::vector<float> result = generatedEvent->get_data<float>();
    return result;
}

void DLInference::SetModelGraph(const std::string &aModelGraph) {
    modelGraph = aModelGraph;
}

void DLInference::SetModelRestore(const std::string &aModelRestore) {
    modelRestore = aModelRestore;
}

void DLInference::SetInputNode(const std::string &anInputNode) {
    inputNode = anInputNode;
}

void DLInference::SetExtraInputNode(const std::string &anExtraInputNode) {
    extraInputNode = anExtraInputNode;
}

void DLInference::SetLabelNode(const std::string &aLabelNode) {
    labelNode = aLabelNode;
}

void DLInference::SetOutputNode(const std::string &anOutputNode) {
    outputNode = anOutputNode;
}

// {100,2}
void DLInference::SetInputShape(const std::vector<int64_t> &anInputShape) {
    inputShape = anInputShape;
}

void DLInference::SetExtraInputShape(const std::vector<int64_t> &anExtraInputShape) {
    extraInputShape = anExtraInputShape;
}

// labelShape = {100,10};
void DLInference::SetLabelShape(const std::vector<int64_t> &aLabelShape) {
    labelShape = aLabelShape;
}

void DLInference::SetEnergyValue(const int &anEnergyValue) {
    energyValue = anEnergyValue;
}

void DLInference::SetInputVecNumber(const int &anInputVecNumber) {
    inputVecNumber = anInputVecNumber;
}

void DLInference::SetExtraInputVecNumber(const int &anExtraInputVecNumber) {
    extraInputVecNumber = anExtraInputVecNumber;
}

//////////////

std::string DLInference::GetModelGraph() {
    return modelGraph;
} 

std::string DLInference::GetModelRestore() {
    return modelRestore;
} 

std::string DLInference::GetInputNode() {
    return inputNode;
}

std::string DLInference::GetExtraInputNode() {
    return extraInputNode;
} 

std::string DLInference::GetLabelNode() {
    return labelNode;
}

std::string DLInference::GetOutputNode() {
    return outputNode;
}

std::vector<int64_t> DLInference::GetInputShape() {
    return inputShape;
}

std::vector<int64_t> DLInference::GetExtraInputShape() {
    return extraInputShape;
}

std::vector<int64_t> DLInference::GetLabelShape() {
    return labelShape;
}

int DLInference::GetEnergyValue() {
    return energyValue;
}

int DLInference::GetInputVecNumber() {
    return inputVecNumber;
}

int DLInference::SetExtraInputVecNumber() {
    return extraInputVecNumber;
}



