#ifndef DL_INFERENCE_DLInference_H
#define DL_INFERENCE_DLInference_H

#include <string>
#include <vector>

class DLInference {

public:

    auto ModelBuildUp();
    auto GenerateInputTensor(std::vector<int64_t> shape, auto &model, std::string node);
    auto GenerateLabelsTensor(std::vector<int64_t> shape, std::vector<float> vecFill, auto &model, std::string node);
    auto TensorBuildUpFloat(std::vector<int64_t> shape, auto vecFillNr, auto &model, std::string node);
    auto TensorBuildUpInt(std::vector<int64_t> shape, auto vecFillNr, auto &model, std::string node);
    std::vector<float> Generation();
    void SetModelGraph(const std::string &aModelGraph);
    void SetModelRestore(const std::string &aModelRestore);
    void SetInputNode(const std::string &anInputNode);
    void SetExtraInputNode(const std::string &anExtraInputNode);
    void SetLabelNode(const std::string &aLabelNode);
    void SetOutputNode(const std::string &anOutputNode);
    void SetInputShape(const  std::vector<int64_t> &anInputShape);
    void SetExtraInputShape(const  std::vector<int64_t> &anExtraInputShape);
    void SetLabelShape(const std::vector<int64_t> &aLabelShape);
    void SetEnergyValue(const std::vector<float> &anEnergyValue);
    void SetInputVecNumber(const float &anInputVecNumber);
    void SetExtraInputVecNumber(const float &anExtraInputVecNumber);
    std::string GetModelGraph();
    std::string GetModelRestore();
    std::string GetInputNode();
    std::string GetExtraInputNode();
    std::string GetLabelNode();
    std::string GetOutputNode();
    std::vector<int64_t> GetInputShape();
    std::vector<int64_t> GetExtraInputShape();
    std::vector<int64_t> GetLabelShape();
    std::vector<float> GetEnergyValue();
    float GetInputVecNumber();
    float SetExtraInputVecNumber();

private:

    std::vector<float> energyValue;
    std::string modelType;
    std::string modelGraph;
    std::string modelRestore;
    std::string inputNode;
    std::string extraInputNode = "";
    std::string labelNode;
    std::string outputNode;

    std::vector<int64_t> inputShape;
    std::vector<int64_t> extraInputShape;
    std::vector<int64_t> labelShape;
    float inputVecNumber;
    float extraInputVecNumber;

};



#endif //DL_INFERENCE_DLInference_H


