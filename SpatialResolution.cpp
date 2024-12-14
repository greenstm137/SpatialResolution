#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <complex>
#include <algorithm>
#include "FourierLib/FFT.h"
#include "FourierLib/Tools.h"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>



int main() {

    void TSE_PD(std::shared_ptr<std::vector<std::complex<float>>>&data, int L, int W, int ETL, float TE, float T2);
    void TSE_T2(std::shared_ptr<std::vector<std::complex<float>>>&data, int L, int W, int ETL, float TE, float T2);
    std::ifstream inputFile("MRIValueTest.txt");
    std::string line;
    std::vector<std::vector<float>> values; // или int, float, string - в зависимости от типа данных
    std::vector<float> InLinevalues;
    if (inputFile.is_open()) {
        while (std::getline(inputFile, line)) 
        {
            if (line == "")
                continue;
            std::stringstream ss(line);
            std::string value;
            while (ss >> value) 
            {
                try 
                {
                    // Преобразование строки в число.  Обработка исключений важна!
                    InLinevalues.push_back(std::stod(value)); // для double
                    // values.push_back(std::stoi(value)); // для int
                    // values.push_back(std::stof(value)); // для float
                }
                catch (const std::invalid_argument& e) 
                {
                    std::cerr << "Ошибка преобразования: " << value << " - " << e.what() << std::endl;
                    // можно обработать ошибку по-другому, например, пропустить значение или завершить работу
                }
            }
            values.push_back(InLinevalues);
            InLinevalues.clear();
        }
        inputFile.close();
    }
    else {
        std::cerr << "Не удалось открыть файл." << std::endl;
    }

    int L = values[0].size(), W = values.size();

    auto data = std::make_shared<std::vector<std::complex<float>>>(values.size() * values[0].size());
    int c = 0;
    for (int i = L - 1; i >= 0; i--)
        for (int j = 0; j < W; j++)
        {
            data->at(c) = std::complex<float>(values[j][i], 0);
            c++;
        }
    int newL = 0 , newW = 0, startrow = 0, startcol = 0;
    auto radix2Data = Tools::ZeroPadding(data, L, W, newL, newW, startrow, startcol);
    uint8_t log2n1 = (uint8_t)log2(newL), log2n2 = (uint8_t)log2(newW);
    std::vector<uint8_t> log2ns = { log2n1, log2n2};

    Tools::Centerize2D(radix2Data, newL, newW);
    FFT::MultiFft(radix2Data, log2ns);
    
    float T2 = 200, TE = 100; //miliseconds
    int ETL = 11;
    std::string filename = "EPI-SSFSE-ETL_" + std::to_string(ETL) + "_TE_" + std::to_string(TE).substr(0, std::to_string(TE).find(".") + 3) + ".png";
    TSE_PD(radix2Data, newL, newW, ETL, TE, T2);
    cv::Mat K_image(newL, newW, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < newL; i++)
        for (int j = 0; j < newW; j++)
            K_image.at<uchar>(i, j) = (uchar)(255 - 25 * log(abs(radix2Data->at(i * newW + j) + 1.0f)));
    bool Ksaved = cv::imwrite("K-space.png", K_image); // Сохранение как PNG
    FFT::InverseMultiFft(radix2Data, log2ns);
    Tools::Centerize2D(radix2Data, newL, newW);
    cv::Mat image(L, W, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < L; i++)
        for (int j = 0; j < W; j++)
            image.at<uchar>(i, j) = (uchar)radix2Data->at((i + startrow) * newW + j + startcol).real();
    cv::transpose(image, image); // Транспонируем изображение
    cv::flip(image, image, 1); // Отзеркаливаем по вертикали
    bool saved = cv::imwrite(filename, image); // Сохранение как PNG
    if (saved) {
        std::cout << "Image sucsessfully saved as output.png";
    }
    else {
        std::cerr << "Error during saving.";
    }
    return 0;
}

void TSE_PD(std::shared_ptr<std::vector<std::complex<float>>>& data, int L, int W, int ETL, float TE, float T2)
{
    int splitsize = L / ETL;
    int remaining = L % ETL;
    std::vector<int> add(ETL, 0);
    for (int i = remaining; i > 0; i--)
        add[i] = 1;
    bool isOdd = (ETL % 2) == 1;
    std::vector<float> decay(ETL);
    int k = 0;
    if (isOdd)      // i.e. ETL = 5:  e^(-TE/T2) = 1/2 => {1/4; 1/2; 1; 1/2; 1/4}
    {
        for (int i = 0; i < ETL / 2; i++)
            decay[i] = 1 / powf(M_E, (TE * (ETL / 2 - i) / T2));
        for (int i = ETL - 1; i > ETL / 2; i--)
            decay[i] = decay[k++];
        decay[ETL / 2] = 1.0f;
    }
    else            // i.e. ETL = 4:  e^(-TE/T2) = 1/2 => {1/4; 1/2; 1; 1/2}
    {
        k++;
        for (int i = 0; i < ETL / 2; i++)
            decay[i] = 1 / powf(M_E, (TE * (ETL / 2 - i) / T2));
        for (int i = ETL - 1; i > ETL / 2; i--)
            decay[i] = decay[k++];
        decay[ETL / 2] = 1.0f;
    }
    int start = 0;
    for (int i = 0; i < ETL; i++)
    {
        for (int j = i * (splitsize + add[i]); j < (splitsize + add[i]) * (i + 1); j++)
        {
            for (k = 0; k < W; k++)
                data->at(j * W + k) *= decay[i];
        }
        start = splitsize * (i + 1) + add[i];
    }
}

void TSE_T2(std::shared_ptr<std::vector<std::complex<float>>>& data, int L, int W, int ETL, float TE, float T2)
{
    int splitsize = L / ETL;
    int remaining = L % ETL;
    std::vector<int> add(ETL, 0);
    for (int i = remaining; i > 0; i--)
        add[i] = 1;
    bool isOdd = (ETL % 2) == 1;
    std::vector<float> decay(ETL);
    int k = 0;
    if (isOdd)      // i.e. ETL = 5:  e^(-TE/T2) = 1/2 => {1; 1/2; 1/4; 1/2; 1}
    {
        for (int i = 0; i < ETL / 2 + 1; i++)
            decay[i] = 1 / powf(M_E, (TE * i / T2));
        for (int i = ETL - 1; i > ETL / 2; i--)
            decay[i] = decay[k++];
    }
    else            // i.e. ETL = 4:  e^(-TE/T2) = 1/2 => {1; 1/2; 1/2; 1}
    {
        for (int i = 0; i < ETL / 2; i++)
            decay[i] = 1 / powf(M_E, (TE * i / T2));
        for (int i = ETL - 1; i >= ETL / 2; i--)
            decay[i] = decay[k++];
    }
    int start = 0;
    for (int i = 0; i < ETL; i++)
    {
        for (int j = start; j < splitsize * (i + 1) + add[i]; j++)
        {
            for (k = 0; k < W; k++)
                data->at(j * W + k) *= decay[i];
        }
        start = splitsize * (i + 1) + add[i];
    }
}