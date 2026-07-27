#include <iostream>
#include "NvInfer.h"

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

int main() {
    std::cout << "TensorRT Version: " << NV_TENSORRT_VERSION << std::endl;
    auto builder = nvinfer1::createInferBuilder(gLogger);
    if (builder) {
        std::cout << "Successfully created TensorRT Builder!" << std::endl;
        delete builder;
    } else {
        std::cout << "Failed to create TensorRT Builder." << std::endl;
    }
    return 0;
}
