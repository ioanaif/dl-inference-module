#ifndef DL_INFERENCE_DLInference_H
#define DL_INFERENCE_DLInference_H

#include <string>
#include <vector>

class DLInference {

public:

    std::vector<float> generation();
    void SetModelGraph(const std::string &aModelGraph);
    void SetModelRestore(const std::string &aModelRestore);
    void SetInputNode(const std::string &anInputNode);
    void SetLabelNode(const std::string &aLabelNode);
    void SetOutputNode(const std::string &anOutputNode);
    void SetInputShape(const  std::vector<int64_t> &anInputShape);
    void SetLabelShape(const std::vector<int64_t> &aLabelShape);
    void SetEnergyValue(const int &anEnergyValue);
    void SetInputVecNumber(const int &anInputVecNumber);

private:

    std::string modelType;
    int energyValue;
    std::string modelGraph;
    std::string modelRestore;
    std::string inputNode;
    std::string labelNode;
    std::string xiNode;
    std::string outputNode;

    int inputSize;
    int labelSize;
    int inputVecNumber;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> labelShape;

    std::string outFileName;

};



#endif //DL_INFERENCE_DLInference_H

