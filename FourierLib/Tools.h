#pragma once

#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_DEPRECATE
#include <cmath>
#include <array>
#include <cstdint>
#include <ppl.h>
#include <concrtrm.h>
#include <iostream>
#include <memory>
#include <vector>
#include <complex>
#include <fstream>
#include <filesystem>
#include <random>
#include <fstream>

using namespace concurrency;

class Tools {
public:
    static std::shared_ptr<std::vector<std::complex<float>>> ZeroPadding(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, int w, int h);

    static std::shared_ptr<std::vector<std::complex<float>>> ZeroPadding(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, int w, int& newL, int& newW, int& startRow, int& startCol);

    static int NextPowOf2(int l);

    static void Centerize3D(std::shared_ptr<std::vector<std::complex<float>>>& data, int l);
    
    static void Centerize2D(std::shared_ptr<std::vector<std::complex<float>>>& data, int L, int W);

    static float EstimateSigmaUsingMad(const std::shared_ptr<std::vector<float>>& data);

    static float GetMedian(const std::vector<float>& data);

    static float CalculateDonohoJohnstoneThreshold(int n, float sigma);

    static float Batterwort(float r, float D0, int n);

    static float Gauss(float r, float D0);

    static float Dist(int x, int y, int z);

    static float WGN(float mean, float sigma);

    static void PrintFull(std:: string filename, std::shared_ptr<std::vector<std::complex<float>>>& data, int l, std::string message);

    static void PrintFullFreq(std::string filename, std::shared_ptr<std::vector<std::complex<float>>>& data, int l, std::string message);

    static void PrintCentralProfile(std::string filename, std::shared_ptr<std::vector<std::complex<float>>>& data, int l, std::string message);

    static float stDev(std::shared_ptr<std::vector<std::complex<float>>>& data, std::shared_ptr<std::vector<std::complex<float>>>& cleardata, int l);

    static float stDev(std::shared_ptr<std::vector<std::complex<float>>>& data, std::shared_ptr<std::vector<std::complex<float>>>& cleardata, int l, float th);

    static float stDevLower(std::shared_ptr<std::vector<std::complex<float>>>& data, std::shared_ptr<std::vector<std::complex<float>>>& cleardata, int l, float th);

    static float Mean(std::shared_ptr<std::vector<std::complex<float>>>& data, int l);

    static float Mean(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, float th);

    static float MeanLower(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, float th);

    static float RSD(float mean, float StD);
};