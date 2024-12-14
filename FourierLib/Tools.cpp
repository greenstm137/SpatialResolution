#include "pch.h"
#include "Tools.h"

std::shared_ptr<std::vector<std::complex<float>>> Tools::ZeroPadding(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, int w, int h)
{
    int L = NextPowOf2(l), W = NextPowOf2(w), H = NextPowOf2(h);
    if (L == l && W == w && H == h)
        return data;
    auto newdata = std::make_shared<std::vector<std::complex<float>>>(L * W * H, std::complex<float>(0.0f, 0.0f));
    std::copy(data->begin(), data->end(), newdata->begin());
    return newdata;
}

std::shared_ptr<std::vector<std::complex<float>>> Tools::ZeroPadding(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, int w, int& newL,
    int& newW, int& startRow, int& startCol)
{
    newL = NextPowOf2(l), newW = NextPowOf2(w);
    if (newL == l && newW == w)
        return data;
    auto newdata = std::make_shared<std::vector<std::complex<float>>>(newL * newW, std::complex<float>(0.0f, 0.0f));

    startRow = (newL - l) / 2;
    startCol = (newW - w) / 2;

    for (int i = 0; i < l; ++i) {
        for (int j = 0; j < w; ++j) {
            newdata->at((i + startRow) * newW + (j + startCol)) = data->at(i * w + j);
        }
    }
    return newdata;
}

int Tools::NextPowOf2(int l)
{
    if (l <= 0)
        throw std::exception("One of dimensions of array is less or equal to zero.");
    l--;
    l |= l >> 1;
    l |= l >> 2;
    l |= l >> 4;
    l |= l >> 8;
    l |= l >> 16;
    return l + 1;
}

void Tools::Centerize3D(std::shared_ptr<std::vector<std::complex<float>>>& data, int l)
{
    parallel_for(0, l, [l, &data](int i)
    {
        for (int j = 0; j < l; j++)
            for (int k = 0; k < l; k++)
                if ((i + j + k) % 2 != 0)
                    data->at(i * l * l + j * l + k) *= -1.0f;
    });
}

void Tools::Centerize2D(std::shared_ptr<std::vector<std::complex<float>>>& data, int L, int W)
{
    for (int i = 0; i < L; i++)
        for (int j = 0; j < W; j++)
            if ((i + j) % 2 != 0)
                data->at(i * W + j) *= -1.0f;
}

float Tools::EstimateSigmaUsingMad(const std::shared_ptr<std::vector<float>>& data)
{
    float median = GetMedian(*data);

    std::vector<float> absoluteDeviations;
    absoluteDeviations.reserve(data->size());
    std::transform(data->begin(), data->end(), std::back_inserter(absoluteDeviations),
        [&](float x) { return std::abs(x - median); });


    float mad = GetMedian(absoluteDeviations);
    return mad / 0.6745;
    
}

float Tools::GetMedian(const std::vector<float>& data)
{
    if (data.empty()) {
        throw std::runtime_error("Median of empty array not defined.");
    }

    std::vector<float> sortedNumbers = data;
    std::sort(sortedNumbers.begin(), sortedNumbers.end());

    size_t size = sortedNumbers.size();
    size_t mid = size / 2;

    if (size % 2 != 0) {
        return sortedNumbers[mid];
    }
    else {
        return (sortedNumbers[mid - 1] + sortedNumbers[mid]) / 2.0;
    }
}

float Tools::CalculateDonohoJohnstoneThreshold(int n, float sigma)
{
    return sigma * sqrtf(2 * logf(n));
}

float Tools::Batterwort(float r, float D0, int n)
{
    float result = 1 / (1 + powf((r / D0), 2 * n));
    return result;
}

float Tools::Gauss(float r, float D0)
{
    double result = 1 / powf(M_E, (r * r) / (2 * D0 * D0));
    return result;
}

float Tools::Dist(int x, int y, int z)
{
    return sqrtf(x * x + y * y + z * z);
}

float Tools::WGN(float mean, float sigma)
{
    std::random_device rd{};
    std::mt19937 gen{ rd() }; // Стандартный генератор Mersenne Twister

    std::normal_distribution<> d(0, 1); // mean = 0, standard deviation = 1

    double z = d(gen);

    double x = mean + z * sigma;
    return (float)x;
}

void Tools::PrintFull(std::string filename, std::shared_ptr<std::vector<std::complex<float>>>& data, int l, std::string message)
{
    FILE* text = fopen(filename.c_str(), "w");
    fprintf(text, "%s\n\n", message.c_str());
    for(int i = 0; i < l; i++)
    {
        fprintf(text, "Silce #%d: \n", i);
            for (int j = 0; j < l; j++)
            {
                for (int k = 0; k < l; k++)
                {
                    fprintf(text, "%.3f\t", data->at(i * l * l + j * l + k).real());
                }
                fprintf(text, "\n");
            }
            fprintf(text, "\n\n");
    }
    fclose(text);
}

void Tools::PrintFullFreq(std::string filename, std::shared_ptr<std::vector<std::complex<float>>>& data, int l, std::string message)
{
    FILE* text = fopen(filename.c_str(), "w");
    fprintf(text, "%s\n\n", message.c_str());
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < l; j++)
        {
            for (int k = 0; k < l; k++)
            {
                fprintf(text, "%5.2f\t", abs(data->at(i * l * l + j * l + k)));
            }
            fprintf(text, "\n");
        }
        fprintf(text, "\n\n");
    }
    fclose(text);
}

void Tools::PrintCentralProfile(std::string filename, std::shared_ptr<std::vector<std::complex<float>>>& data, int l, std::string message)
{
    FILE* text = fopen(filename.c_str(), "a+");
    fprintf(text, "%s\n\n", message.c_str());
    int i = l / 2, j = l / 2;
    for (int k = 0; k < l; k++)
    {              
        fprintf(text, "%d\t%5.2f\n", k, abs(data->at(i * l * l + j * l + k)));
    }
    fprintf(text, "\n");

    fclose(text);
}

float Tools::stDev(std::shared_ptr<std::vector<std::complex<float>>>& data, std::shared_ptr<std::vector<std::complex<float>>>& cleardata, int l)
{
    std::vector<float>sqd(l);
    double diff = 0;
    parallel_for(0, l, [&](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                    sqd[i] += (data->at(i * l * l + j * l + k) - cleardata->at(i * l * l + j * l + k)).real()
                    * (data->at(i * l * l + j * l + k) - cleardata->at(i * l * l + j * l + k)).real();
        });
    for (const auto& s : sqd)
        diff += s;
    return sqrtf(diff / (l * l * l - 1));
}

float Tools::stDev(std::shared_ptr<std::vector<std::complex<float>>>& data, std::shared_ptr<std::vector<std::complex<float>>>& cleardata, int l, float th)
{
    std::vector<float>sqd(l);
    double diff = 0;
    std::vector<int>count(l);
    int C = 0;
    parallel_for(0, l, [&](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                    if (cleardata->at(i * l * l + j * l + k).real() >= th)
                    {
                        sqd[i] += (data->at(i * l * l + j * l + k) - cleardata->at(i * l * l + j * l + k)).real()
                            * (data->at(i * l * l + j * l + k) - cleardata->at(i * l * l + j * l + k)).real();
                        count[i]++;
                    }
        });
    for (const auto& s : sqd)
        diff += s;
    for (const auto& c : count)
        C += c;
    return sqrtf(diff / (C - 1));
}

float Tools::stDevLower(std::shared_ptr<std::vector<std::complex<float>>>& data, std::shared_ptr<std::vector<std::complex<float>>>& cleardata, int l, float th)
{
    std::vector<float>sqd(l);
    double diff = 0;
    std::vector<int>count(l);
    int C = 0;
    parallel_for(0, l, [&](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                    if (cleardata->at(i * l * l + j * l + k).real() <= th)
                    {
                        sqd[i] += (data->at(i * l * l + j * l + k) - cleardata->at(i * l * l + j * l + k)).real()
                            * (data->at(i * l * l + j * l + k) - cleardata->at(i * l * l + j * l + k)).real();
                        count[i]++;
                    }
        });
    for (const auto& s : sqd)
        diff += s;
    for (const auto& c : count)
        C += c;
    return sqrtf(diff / (C - 1));
}

float Tools::Mean(std::shared_ptr<std::vector<std::complex<float>>>& data, int l)
{
    std::vector<float>mean(l);
    float M = 0;
    parallel_for(0, l, [&mean, l, &data](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                    mean[i] += data->at(i * l * l + j * l + k).real();
        });
    for (const auto& m : mean)
        M += m;
    return M / l / l / l;
}

float Tools::Mean(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, float th)
{
    std::vector<float>mean(l);
    std::vector<int>count(l);
    float M = 0;
    int C = 0;
    parallel_for(0, l, [&mean, &count, l, th, &data](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                    if (data->at(i * l * l + j * l + k).real() >= th)
                    {
                        mean[i] += data->at(i * l * l + j * l + k).real();
                        count[i]++;
                    }
        });
    for (const auto& m : mean)
        M += m;
    for (const auto& c : count)
        C += c;
    return M / C;
}

float Tools::MeanLower(std::shared_ptr<std::vector<std::complex<float>>>& data, int l, float th)
{
    std::vector<float>mean(l);
    std::vector<int>count(l);
    float M = 0;
    int C = 0;
    parallel_for(0, l, [&mean, &count, l, th, &data](int i)
        {
            for (int j = 0; j < l; j++)
                for (int k = 0; k < l; k++)
                    if (data->at(i * l * l + j * l + k).real() <= th)
                    {
                        mean[i] += data->at(i * l * l + j * l + k).real();
                        count[i]++;
                    }
        });
    for (const auto& m : mean)
        M += m;
    for (const auto& c : count)
        C += c;
    return M / C;
}

float Tools::RSD(float mean, float StD)
{
    float rsd = 0;
    rsd = StD / mean;
    return rsd;
}




