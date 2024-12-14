#pragma once

#define _USE_MATH_DEFINES
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
#include "Tools.h"

using namespace concurrency;

//Cooley-Tukey algorithm  
//details - https://ru.dsplib.org/content/fft_dec_in_time/fft_dec_in_time.html
class FFT
{
public:
    /*
    * Multi-dimensional FFT using Cooley-Tukey radix 2 decimation-in-time algorithm. Data - K*L*...*T array and log2ns = {log2(K); log2(L); ... ; log2(T)}
    */
    static void MultiFft(std::shared_ptr<std::vector<std::complex<float>>>& data, std::vector<uint8_t> log2ns);
    /*
    * Inverse MultyFFT. Parameters are the same as MultiFFT;
    */
    static void InverseMultiFft(std::shared_ptr<std::vector<std::complex<float>>>& data, std::vector<uint8_t> log2ns);
    /*
    * Function return reverse bits of index n with bitCount. Helpful function to find odd/even split for FFT
    */
    static int ReverseBits(int n, int bitCount);
    /*
    * Gaussian filter function with parameters D0 - filter cutoff frequency, data - array, l - size of one-dimension.
    * 
    * WORKS ONLY WITH l*l*l array
    */
    static void GaussFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int l);
    /*
    * Gaussian filter function with parameters D0 - filter cutoff frequency, data - array, l - size of one-dimension.
    *
    * WORKS ONLY WITH l*l*l array
    * 
    * Recommended ThMultiplier = 1.66
    */
    static void GaussFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int l, float ThMultiplier);
    /*
    * Batterwort filter function with parameters D0 - filter cutoff frequency, n - filter order, data - array, l - size of one-dimension.
    *
    * WORKS ONLY WITH l*l*l array
    */
    static void BatterwortFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int n, int l);
    /*
    * Batterwort filter function with parameters D0 - filter cutoff frequency, n - filter order, data - array, l - size of one-dimension.
    *
    * WORKS ONLY WITH l*l*l array
    * 
    * Recommended ThMultiplier = 1.66
    */
    static void BatterwortFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int n, int l, float ThMultiplier);

    static void BatterwortFilter2D(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int n, int l, float ThMultiplier);

private:
    /*
    * Function realising FFT through one of dimensions. For example we have 3D-array [512] <=> [8][8][8]
    * This function applies to [i][][] elements at first, then to [][i][] elements and, finally to [][][i]
    * The results in "data" are equivalent to 3D-DFT
    */
    static void FftMultiDPlane(std::shared_ptr<std::vector<std::complex<float>>>& data, int planeShift, int plane, int log2N);
    /*
    * Function realising back-reverse of indexes after FFT
    */
    static void ReverseMultiDPlane(std::shared_ptr<std::vector<std::complex<float>>>& data, int plane, int len, int stride);

    static std::array<uint8_t, 256> generate_byte_rbits();
    /*
    * Pre-calculated reverse bits
    */
    static std::array<uint8_t, 256> BYTE_RBITS;

    static std::array<std::complex<float>, 32> ComputePow2RootsOfUnity();
    /*
    * Pre-calculated power of 2 roots of unity <=> (-1)^(1/n)
    */
    static std::array<std::complex<float>, 32> POW2_UNITY_ROOT;

};