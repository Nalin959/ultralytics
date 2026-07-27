#include <NvInfer.h>
#include <iostream>
#include <fstream>
using namespace std;
using namespace nvinfer1;
class Logger : public ILogger { void log(Severity severity, const char* msg) noexcept override {} } gLogger;
int main() {
    ifstream file("lightglue.engine", ios::binary);
    if (!file) { cerr << "No file!" << endl; return 1; }
    file.seekg(0, ios::end); size_t size = file.tellg(); file.seekg(0, ios::beg);
    char* buf = new char[size]; file.read(buf, size);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buf, size);
    for(int i=0; i<engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        Dims dims = engine->getTensorShape(name);
        cout << name << ": ";
        for(int j=0; j<dims.nbDims; j++) cout << dims.d[j] << " ";
        cout << endl;
        if(engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            Dims minD = engine->getProfileShape(name, 0, OptProfileSelector::kMIN);
            Dims optD = engine->getProfileShape(name, 0, OptProfileSelector::kOPT);
            Dims maxD = engine->getProfileShape(name, 0, OptProfileSelector::kMAX);
            cout << "  MIN: "; for(int j=0; j<minD.nbDims; j++) cout << minD.d[j] << " "; cout << endl;
            cout << "  OPT: "; for(int j=0; j<optD.nbDims; j++) cout << optD.d[j] << " "; cout << endl;
            cout << "  MAX: "; for(int j=0; j<maxD.nbDims; j++) cout << maxD.d[j] << " "; cout << endl;
        }
    }
    return 0;
}
