#include "pch.h"
#include "FFT.h"

void FFT::MultiFft(std::shared_ptr<std::vector<std::complex<float>>>& data, std::vector<uint8_t> log2ns)
{
    int dn = log2ns.size();
    int totalbits = 0;
    for (const auto& b : log2ns) {
        totalbits += (int)b;
    }
    if (data->size() < (1 << totalbits))
    {
        throw std::exception("dimensional mismatch");
    }
    int lowerBits = 0, lowerMask = 0; //lower bits and lower mask relative to current dimension
    for (int d = 0; d < dn; ++d)    //going through dimensions
    {
        int log2n = log2ns[dn - d - 1]; //log2 of current dim size

        int shift = lowerBits;  //Necessary shift for going throug current dim

        int nSize = 1 << (totalbits - log2n);   //total iterations through dimension
        int upperMask = ~lowerMask << log2n; //upper mask for current dimension
        concurrency::parallel_for(0, nSize, [lowerMask, log2n, upperMask, shift, &data](int i) 
        {
            int plane = (i & lowerMask) | (i << log2n) & upperMask; //calculating plane start for each i

            FftMultiDPlane(data, shift, plane, log2n);  //Calculating FFT for current plane
        });
        lowerBits += log2n; //updating lower bits and lower mask for next dim
        lowerMask = ~(~lowerMask << log2n);
    }
}

void FFT::InverseMultiFft(std::shared_ptr<std::vector<std::complex<float>>>& data, std::vector<uint8_t> log2ns)
{
    short dn = log2ns.size();
    int totalbits = 0;
    for (const auto& b : log2ns) {
        totalbits += (int)b;
    }
    if (data->size() < (1 << totalbits))
    {
        throw std::exception("dimensional mismatch");
    }
    int lowerBits = 0, lowerMask = 0;
    for (int d = 0; d < dn; ++d)
    {
        uint8_t log2n = log2ns[dn - d - 1];
        int n = 1 << log2n;

        int shift = lowerBits;

        int nSize = 1 << (totalbits - log2n);
        int upperMask = ~lowerMask << log2n;

        int stride = 1 << lowerBits;

        concurrency::parallel_for(0, nSize, [lowerMask, log2n, upperMask, shift, n, stride, &data](int i) 
        {
            int plane = (i & lowerMask) | (i << log2n) & upperMask;
            ReverseMultiDPlane(data, plane + stride, n - 1, stride);
            FftMultiDPlane(data, shift, plane, log2n);
        });

        lowerBits += log2n;
        lowerMask = ~(~lowerMask << log2n);
    }
    
    int totalLen = data->size();
    concurrency::parallel_for(0, totalLen, [&data, totalLen](int i) 
    {
        data->at(i) /= totalLen;
    });
}

int FFT::ReverseBits(int n, int bitCount)
{
    int r = n << (32 - bitCount);
    int result = (BYTE_RBITS[r & 0xFF] << 24)
        | (BYTE_RBITS[(r >> 8) & 0xFF] << 16)
        | (BYTE_RBITS[(r >> 16) & 0xFF] << 8)
        | BYTE_RBITS[(r >> 24) & 0xFF];
    return result;
}

void FFT::GaussFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int l)
{
    auto Magnitudes = std::make_shared<std::vector<float>>(data->size());
    std::transform(data->begin(), data->end(), Magnitudes->begin(),
        [](const std::complex<float>& c) { return abs(c); });
    float Nsigma = Tools::EstimateSigmaUsingMad(Magnitudes);
    float threshold = 1.66 * Tools::CalculateDonohoJohnstoneThreshold(l * l * l, Nsigma);
    concurrency::parallel_for(0, l, [l, &data, D0, threshold](int i)
    {
        for (int j = 0; j < l; j++)
            for (int k = 0; k < l; k++)
            {
                int x = i - l / 2, y = j - l / 2, z = k - l / 2;
                double r = Tools::Dist(x, y, z);
                data->at(i * l * l + j * l + k) = abs(data->at(i * l * l + j * l + k)) < threshold ?
                    data->at(i * l * l + j * l + k) * Tools::Gauss(r, D0) : data->at(i * l * l + j * l + k);
            }
    });
}

void FFT::GaussFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int l, float ThMultiplier)
{
    auto Magnitudes = std::make_shared<std::vector<float>>(data->size());
    std::transform(data->begin(), data->end(), Magnitudes->begin(),
        [](const std::complex<float>& c) { return abs(c); });
    float Nsigma = Tools::EstimateSigmaUsingMad(Magnitudes);
    float threshold = ThMultiplier * Tools::CalculateDonohoJohnstoneThreshold(l * l * l, Nsigma);
    concurrency::parallel_for(0, l, [l, &data, D0, threshold](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                {
                    int x = i - l / 2, y = j - l / 2, z = k - l / 2;
                    double r = Tools::Dist(x, y, z);
                    data->at(i * l * l + j * l + k) = abs(data->at(i * l * l + j * l + k)) < threshold ?
                        data->at(i * l * l + j * l + k) * Tools::Gauss(r, D0) : data->at(i * l * l + j * l + k);
                }
        });
}

void FFT::BatterwortFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int n, int l)
{
    auto Magnitudes = std::make_shared<std::vector<float>>(data->size());
    std::transform(data->begin(), data->end(), Magnitudes->begin(),
        [](const std::complex<float>& c) { return abs(c); });
    float Nsigma = Tools::EstimateSigmaUsingMad(Magnitudes);
    float threshold = 1.66 * Tools::CalculateDonohoJohnstoneThreshold(l * l * l, Nsigma);
    concurrency::parallel_for(0, l, [l, n, &data, D0, threshold](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                {
                    int x = i - l / 2, y = j - l / 2, z = k - l / 2;
                    double r = Tools::Dist(x, y, z);
                    data->at(i * l * l + j * l + k) = abs(data->at(i * l * l + j * l + k)) < threshold ?
                        data->at(i * l * l + j * l + k) * Tools::Batterwort(r, D0, n) : data->at(i * l * l + j * l + k);
                }
        });
}

void FFT::BatterwortFilter(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int n, int l, float ThMultiplier)
{
    auto Magnitudes = std::make_shared<std::vector<float>>(data->size());
    std::transform(data->begin(), data->end(), Magnitudes->begin(),
        [](const std::complex<float>& c) { return abs(c); });
    float Nsigma = Tools::EstimateSigmaUsingMad(Magnitudes);
    float threshold = ThMultiplier * Tools::CalculateDonohoJohnstoneThreshold(l * l * l, Nsigma);
    concurrency::parallel_for(0, l, [l, n, &data, D0, threshold](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                {
                    int x = i - l / 2, y = j - l / 2, z = k - l / 2;
                    double r = Tools::Dist(x, y, z);
                    data->at(i * l * l + j * l + k) = abs(data->at(i * l * l + j * l + k)) < threshold ?
                        data->at(i * l * l + j * l + k) * Tools::Batterwort(r, D0, n) : data->at(i * l * l + j * l + k);
                }
        });
}

void FFT::BatterwortFilter2D(std::shared_ptr<std::vector<std::complex<float>>>& data, float D0, int n, int l, float ThMultiplier)
{
    auto Magnitudes = std::make_shared<std::vector<float>>(data->size());
    std::transform(data->begin(), data->end(), Magnitudes->begin(),
        [](const std::complex<float>& c) { return abs(c); });
    float Nsigma = Tools::EstimateSigmaUsingMad(Magnitudes);
    float threshold = ThMultiplier * Tools::CalculateDonohoJohnstoneThreshold(l * l * l, Nsigma);
    for (int j = 0; j < l; j++)
        for (int k = 0; k < l; k++)
        {
            int x = j - l / 2, y = k - l / 2;
            double r = Tools::Dist(x, y, 0);
            data->at(j * l + k) = abs(data->at(j * l + k)) < threshold ?
            data->at(j * l + k) * Tools::Batterwort(r, D0, n) : data->at(j * l + k);
        }
}

void FFT::FftMultiDPlane(std::shared_ptr<std::vector<std::complex<float>>>& data, int planeShift, int plane, int log2N)
{
    int dp = 1 << planeShift;   //step between planes
    int n = 1 << log2N; //one block for reverse bits of indexes for FFT
    int ipos = plane;
    for (int i = 0; i < n; ++i, ipos += dp) //sqap indexes for FFT using reverse bits   
    {
        int r = ReverseBits(i, log2N);
        if (i >= r) continue;

        int rpos = plane + (r << planeShift);
        std::swap(data->at(ipos), data->at(rpos));
    }

    int s = 0;
    int pmax = 1 << (log2N + planeShift);   //max index for current dim
    int kmax = plane + pmax;    //final position for current dim
    int dk = dp + dp;
    for (int dj = dp; dj < pmax; dj = dk, dk += dk)     //Cooley-Tukey algorithm realisation 
                                                        //(details - https://ru.dsplib.org/content/fft_dec_in_time/fft_dec_in_time.html)
    {
        std::complex<float> dw = POW2_UNITY_ROOT[s++];
        for (int k0 = plane; k0 < kmax; k0 += dk)
        {
            int k1 = k0 + dj;

            std::complex<float> x0 = data->at(k0);
            std::complex<float> x1 = data->at(k1);

            data->at(k0) = x0 + x1; //2-point DFT
            data->at(k1) = x0 - x1;

            std::complex<float> w = dw;
            int jmax = k0 + dj;
            for (int j0 = k0 + dp; j0 < jmax; w *= dw, j0 += dp)
            {
                int j1 = j0 + dj;

                x0 = data->at(j0);
                x1 = w * data->at(j1);

                data->at(j0) = x0 + x1; //2-point DFT
                data->at(j1) = x0 - x1;
            }
        }
    }
}

void FFT::ReverseMultiDPlane(std::shared_ptr<std::vector<std::complex<float>>>& data, int plane, int len, int stride)
{
    int i = plane;
    int r = plane + (len - 1) * stride;

    while (i < r)
    {
        std::swap(data->at(i), data->at(r));
        i += stride;
        r -= stride;
    }
}

std::array<uint8_t, 256> FFT::generate_byte_rbits()
{
    std::array<uint8_t, 256> rbytes;

    for (int i = 0; i < 256; ++i)
    {
        int fwd = i, bwd = 0;
        for (int bit = 0; bit < 8; ++bit)
        {
            bwd <<= 1;
            bwd |= fwd & 1;
            fwd >>= 1;
        }

        rbytes[i] = (uint8_t)bwd;
    }

    return rbytes;
}

std::array<std::complex<float>, 32> FFT::ComputePow2RootsOfUnity()
{
    const short maxPow2 = 32;
    std::array<std::complex<float>, maxPow2> result;
    result[0] = std::complex<float>(-1, 0);
    std::array<float, maxPow2 - 1> theta;
    theta[0] = (float)M_PI * 0.5;
    for (int i = 1; i < maxPow2 - 1; i++)
        theta[i] = theta[i - 1] * 0.5f;

    for (int i = 1; i < maxPow2; i++)
    {
        float u = cos(theta[i - 1]);
        float v = sin(theta[i - 1]);
        result[i] = std::complex<float>(u, -v);
    }
    return result;
}

std::array<std::complex<float>, 32> FFT::POW2_UNITY_ROOT = FFT::ComputePow2RootsOfUnity();
std::array<uint8_t, 256> FFT::BYTE_RBITS = FFT::generate_byte_rbits();

