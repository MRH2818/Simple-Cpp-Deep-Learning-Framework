#pragma once
#include <fstream>
#include <vector>

namespace deeplframework {
    namespace data {
        class MnistDataReader {
        private:
            std::ifstream labelReader;
            std::ifstream imageReader;
            const char* labelFile;
            const char* imageFile;

        public:
            MnistDataReader(const char* labelFilePath, const char* imageFilePath) {
                this->labelFile = labelFilePath;
                this->imageFile = imageFilePath;
            }

            ~MnistDataReader() {
                this->labelReader.close();
                this->imageReader.close();
            }

            void open() {
                this->labelReader.open(this->labelFile, std::ios::binary);
                this->imageReader.open(this->imageFile, std::ios::binary);
            }

            void close() {
                labelReader.close();
            }

            // LabelsFile must be opened in binary format. Function returns -1 if there is a problem.
            // Function does not close ifstream connection to file.
            std::vector<double> getLabelOutput(int labelId) {
                if (labelReader.is_open()) {
                    labelReader.seekg(labelId + 8);

                    // Get label
                    int label = 0;
                    labelReader.read(reinterpret_cast<char*>(&label), 1);

                    std::vector<double> labelId;

                    // Make output
                    for (int i = 0; i < 10; i++) {
                        labelId.push_back(((i == label) ? 1.0 : 0.0));
                    }
                    return labelId;
                }
                else {
                    throw std::runtime_error("Object is not open");
                }
                return {};
            }

            // Pixel values will range from 0-1, and the image will be squashed onto a single vector, arranged row-wise
            std::vector<double> getImageInput(int imageId) {
                if (imageReader.is_open()) {
                    std::vector<double> image;

                    imageReader.seekg(16 + (imageId * 784));

                    for (int i = 0; i < 784; i++) {
                        int pixelValue = 0;
                        imageReader.read(reinterpret_cast<char*>(&pixelValue), 1);

                        image.push_back((double)pixelValue / 255.0);
                    }

                    return image;
                }
                else {
                    throw std::runtime_error("Object is not open");
                }
                return {};
            }
        };
    }
}