 #include "DLInference.h"
#include "Model.h"
#include "Tensor.h"
#include <typeinfo>


auto DLInference::ModelBuildUp() {
    Model m(modelGraph);
    m.restore(modelRestore);
    return m;
}

float RandomFloat() {
    float r = (float)std::rand() / (float)RAND_MAX;
    return -1 + r * 2;
}

auto DLInference::GenerateInputTensor(std::vector<int64_t> shape, auto &model, std::string node) {
    int size = std::accumulate(begin(shape), end(shape), 1, std::multiplies<>());
    std::vector<float> toVec(size);
    std::generate(toVec.begin(), toVec.end(), RandomFloat);
    Tensor dataTensor{model, node};
    dataTensor.set_data(toVec, shape);
    return dataTensor;
}


auto DLInference::GenerateLabelsTensor(std::vector<int64_t> shape, std::vector<float> vecFill, auto &model, std::string node) {
    int size = std::accumulate(begin(shape), end(shape), 1, std::multiplies<>());
    std::vector<float> toVec(size);
    int times = toVec.size() / vecFill.size();
    for (int i = 0; i < times; i++) {
    	toVec.insert(toVec.begin(), vecFill.begin(), vecFill.end());
    }
    Tensor dataTensor{model, node};
    dataTensor.set_data(toVec, shape);
    return dataTensor;
}


auto DLInference::TensorBuildUpFloat(std::vector<int64_t> shape, auto vecFillNr, auto &model, std::string node) {
   
    int size = std::accumulate(begin(shape), end(shape), 1, std::multiplies<>());
//    std::cout<<typeid(vecFillNr).name() + node<<std::endl; 
//    std::cout<<vecFillNr<<std::endl;
//    if (typeid(vecFillNr) == typeid(int)){
//        std::cout<<"INT*********"<<std::endl;
//        std::vector<int> toVec(size);
//        std::fill(toVec.begin(), toVec.end(), vecFillNr);
//        Tensor dataTensor{model, node};
//        dataTensor.set_data(toVec, shape);
//        return dataTensor;
//    } else {
//        std::cout<<"FLOAT*********"<<std::endl; 
//        std::vector<float> toVec(size);
//        std::fill(toVec.begin(), toVec.end(), vecFillNr);
//        Tensor dataTensor{model, node};
//        dataTensor.set_data(toVec, shape);
//        return dataTensor;
//    }
    std::vector<float> toVec(size);
    std::fill(toVec.begin(), toVec.end(), vecFillNr);
    Tensor dataTensor{model, node};
    dataTensor.set_data(toVec, shape);

    return dataTensor;
}

auto DLInference::TensorBuildUpInt(std::vector<int64_t> shape, auto vecFillNr, auto &model, std::string node) {
   
    int size = std::accumulate(begin(shape), end(shape), 1, std::multiplies<>());
//    std::cout<<typeid(vecFillNr).name() + node<<std::endl; 
//    std::cout<<vecFillNr<<std::endl;
//    if (typeid(vecFillNr) == typeid(int)){
//        std::cout<<"INT*********"<<std::endl;
//        std::vector<int> toVec(size);
//        std::fill(toVec.begin(), toVec.end(), vecFillNr);
//        Tensor dataTensor{model, node};
//        dataTensor.set_data(toVec, shape);
//        return dataTensor;
//    } else {
//        std::cout<<"FLOAT*********"<<std::endl; 
//        std::vector<float> toVec(size);
//        std::fill(toVec.begin(), toVec.end(), vecFillNr);
//        Tensor dataTensor{model, node};
//        dataTensor.set_data(toVec, shape);
//        return dataTensor;
//    }
    std::vector<int> toVec(size);
    std::fill(toVec.begin(), toVec.end(), vecFillNr);
    Tensor dataTensor{model, node};
    dataTensor.set_data(toVec, shape);

    return dataTensor;
}


std::vector<float> DLInference::Generation() {

    Model model = ModelBuildUp();

    auto inputData = GenerateInputTensor(inputShape, model, inputNode);
    auto eventEnergy = GenerateLabelsTensor(labelShape, energyValue, model, labelNode);
    auto generatedEvent = new Tensor(model, outputNode);

    if (extraInputNode != "") {
        auto xInput = GenerateInputTensor(extraInputShape, model, extraInputNode);
        model.run({&xInput,&inputData,&eventEnergy}, generatedEvent);
    } else {
        model.run({&inputData,&eventEnergy}, generatedEvent);
    }

    // Get Generated Event Tensor
    auto result = generatedEvent->get_data<float>();
//    auto result = generatedEvent[0].flat<float>();
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

void DLInference::SetEnergyValue(const std::vector<float> &anEnergyValue) {
    energyValue = anEnergyValue;
}

void DLInference::SetInputVecNumber(const float  &anInputVecNumber) {
    inputVecNumber = anInputVecNumber;
}

void DLInference::SetExtraInputVecNumber(const float &anExtraInputVecNumber) {
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

std::vector<float> DLInference::GetEnergyValue() {
    return energyValue;
}

float DLInference::GetInputVecNumber() {
    return inputVecNumber;
}

float DLInference::SetExtraInputVecNumber() {
    return extraInputVecNumber;
}




