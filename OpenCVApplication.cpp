#include "stdafx.h"
#include "common.h"
#include <random>

float addSaturation(float x, float y)
{
	float z = x + y;
	if (z > 1.0)
		z = 1.0;
	if (z < 0.0)
		z = 0.0;
	return z;
}

bool isInside(Mat img, int i, int j)
{
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
	{
		return true;
	}
	else
	{
		return false;
	}
}

Mat_<float> GaussNoiseF(Mat_<float> inputImage, boolean show)
{
	Mat_<float> noise(inputImage.size(), inputImage.type());
	Mat_<float> noisyImage = inputImage.clone();

	float mean = 0.0f;
	float sigma = 5.01 / 255.0f;
	cv::randn(noise, mean, sigma);
	noisyImage += noise;

	if (show == true)
	{
		imshow("Original Image", inputImage);
		imshow("Noisy Image", noisyImage);
	}

	return noisyImage;
}

float computeMean(float* fdp)
{
	float mean = 0.0;
	for (int i=0; i<256; i++)
	{
		mean += (i * fdp[i]);
	}
	return mean;
}

float computeVariance(float* fdp)
{
	float mean = computeMean(fdp);
	float var = 0.0f;

	for (int i=0; i<256; i++)
	{
		var += (i - mean) * (i - mean) * fdp[i];
	}
	return var;
}

void calculateHistogram(Mat_<float> inputImage, int nrBins, int* hist, float* fdp)
{
	for (int i=0; i<nrBins; i++)
		hist[i] = 0;

	Mat_<uchar> inputNormal;
	inputImage.convertTo(inputNormal, CV_8UC1, 255);

	for (int i=0; i<inputImage.rows; i++)
	{
		for (int j=0; j<inputImage.cols; j++)
		{
			int newValue = (int) inputNormal(i, j);
			hist[newValue]++;
		}
	}

	for (int i=0; i<256; i++)
	{
		fdp[i] = (float) hist[i] / (inputImage.rows * inputImage.cols);
	}
}

float** computeK(Mat_<float> noisyImage, int r, float noiseVariance, float a, float b)
{
	int row = noisyImage.rows;
	int col = noisyImage.cols;

	float** K = (float**) malloc(row * sizeof(float*));
	for (int i=0; i<row; i++)
		K[i] = (float*) malloc(col * sizeof(float));

	float epsilon = b * sqrt(noiseVariance) / 255.0f / 255.0f;
	float weight = 0.0f;
	printf("epsilon = %f\n", epsilon);

	for (int i=0; i<row; i++)
		for (int j=0; j<col; j++)
		{
			float sum = 0.0f;
			for (int u=0; u<r; u++)
			{
				for (int v=0; v<r; v++)
				{
					if (isInside(noisyImage, i+u-r/2, j+v-r/2))
					{
						float D = (noisyImage(i+u-r/2, j+v-r/2) - noisyImage(i, j)) * (noisyImage(i+u-r/2, j+v-r/2) - noisyImage(i, j));
						weight = 1.0f / (1.0f + a * max(epsilon, D));
						sum += weight;
					}
				}
			}

			K[i][j] = 1.0f / sum;
		}
	return K;
}

float** meanSignal(Mat_<float> noisyImage, int r, float noiseVariance, float a, float b, boolean weighted)
{
	int row = noisyImage.rows;
	int col = noisyImage.cols;

	float** meanMatrix = (float**) malloc(row * sizeof(float*));
	for (int i=0; i<row; i++)
		meanMatrix[i] = (float*) malloc(col * sizeof(float));


	float **K = computeK(noisyImage, r, noiseVariance, a, b);
	float epsilon = b * sqrt(noiseVariance) / 255.0f / 255.0f;
	float weight;

	for (int i=0; i<row; i++)
		for (int j=0; j<col; j++)
		{
			float sum = 0.0f;

			for (int u=0; u<r; u++)
			{
				for (int v=0; v<r; v++)
				{
					if (isInside(noisyImage, i+u-r/2, j+v-r/2))
					{
						float D = (noisyImage(i+u-r/2, j+v-r/2) - noisyImage(i, j)) * (noisyImage(i+u-r/2, j+v-r/2) - noisyImage(i, j));

						if (weighted == true)
						{
							weight = 1.0f / (1 + a * max(epsilon, D));
							sum += K[i][j] * weight * noisyImage(i+u-r/2, j+v-r/2);
						}
						else
						{
							sum += noisyImage(i+u-r/2, j+v-r/2);
						}
					}
				}
			}

			if (weighted == true)
				meanMatrix[i][j] = sum;
			else
				meanMatrix[i][j] = sum * 1.0f / (2*r+1) / (2*r+1);
		}
	return meanMatrix;
}

float** varianceSignal(Mat_<float> noisyImage, int r, float noiseVariance, float a, float b, boolean weighted)
{
	int row = noisyImage.rows;
	int col = noisyImage.cols;

	float** varianceMatrix = (float**) malloc(row * sizeof(float*));
	for (int i=0; i<row; i++)
		varianceMatrix[i] = (float*) malloc(col * sizeof(float));

	float** meanMatrix = meanSignal(noisyImage, r, noiseVariance, a, b, weighted);
	float sigma = noiseVariance / 255.0f / 255.0f;

	float** K = computeK(noisyImage, r, noiseVariance, a, b);
	float epsilon = b * sqrt(noiseVariance) / 255.0f / 255.0f;
	float weight;

	for (int i=0; i<row; i++)
		for (int j=0; j<col; j++)
		{
			float sum = 0.0f;
			for (int u=0; u<r; u++)
			{
				for (int v=0; v<r; v++)
				{
					if (isInside(noisyImage, i+u-r/2, j+v-r/2))
					{
						if (weighted == true)
						{
							weight = 1.0f / (1 + a * max(epsilon, (noisyImage(i+u-r/2, j+v-r/2) - noisyImage(i, j)) * (noisyImage(i+u-r/2, j+v-r/2) - noisyImage(i, j))));
							sum += K[i][j] * weight * (noisyImage(i+u-r/2, j+v-r/2) - meanMatrix[i][j]) * (noisyImage(i+u-r/2, j+v-r/2) - meanMatrix[i][j]);
						}
						else
							sum += (noisyImage(i+u-r/2, j+v-r/2) - meanMatrix[i][j]) * (noisyImage(i+u-r/2, j+v-r/2) - meanMatrix[i][j]);
							
					}
				}
			}

			if (weighted == true)
				varianceMatrix[i][j] = sum;
			else
				varianceMatrix[i][j] = sum * 1.0f / (2*r+1) / (2*r+1);;
		}
	return varianceMatrix;
}

Mat_<float> reconstructImage(Mat_<float> noisyImage, int r, float noiseVariance, float a, float b, boolean weighted)
{
	Mat_<float> outputImage = Mat_<float>(noisyImage.rows, noisyImage.cols);
	float** varianceMatrix = varianceSignal(noisyImage, r, noiseVariance, a, b, weighted);
	float** meanMatrix = meanSignal(noisyImage, r, noiseVariance, a, b, weighted);
	float sigma = noiseVariance / 255.0f;

	for (int i=0; i<noisyImage.rows; i++)
	{
		for (int j=0; j<noisyImage.cols; j++)
		{
			outputImage(i, j) = addSaturation(varianceMatrix[i][j] * 1.0f /
				addSaturation(varianceMatrix[i][j], sigma)
				* addSaturation(noisyImage(i, j), (-1.0f) * meanMatrix[i][j]), meanMatrix[i][j]);

			outputImage(i, j) += addSaturation(meanMatrix[i][j], max(0.0f, addSaturation(sigma, -1.0f * varianceMatrix[i][j]))
				/ max(sigma, varianceMatrix[i][j]) * addSaturation(noisyImage(i, j), -1.0f * meanMatrix[i][j]));

			outputImage(i, j) /= 2;
		}	
	}
	return outputImage;
}

float computeMSE(Mat_<float> inputImage, Mat_<float> outputImage)
{
	int N = inputImage.rows;
	int M = inputImage.cols;
	float MSE = 0.0f;

	for(int i=0; i<N; i++)
		for (int j = 0; j<M; j++)
		{
			MSE += (outputImage(i, j) - inputImage(i, j)) * (outputImage(i, j) - inputImage(i, j));
		}

	MSE /= N * M;
	return MSE;
}

float computePSNR(float MSE)
{
	return 10 * log10(1.0f / MSE);
}

int main()
{
	int hist[256];
	float fdp[256];
	int nrBins = 256;
	int r;
	float a, b;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		std::cout << "r = ";
		scanf("%d", &r);

		std::cout << "a = ";
		scanf("%f", &a);

		std::cout << "b = ";
		scanf("%f", &b);

		Mat_<uchar> inputImage = imread(fname, IMREAD_GRAYSCALE);
		Mat_<float> inputImageFloat;

		inputImage.convertTo(inputImageFloat, CV_32FC1, 1 / 255.0);
		Mat_<float> noisyImage = GaussNoiseF(inputImageFloat, true);

		calculateHistogram(noisyImage, nrBins, hist, fdp);

		double t = (double) getTickCount();
		float noiseMedian = computeMean(fdp);
		std::cout << "Noisy Median: " << noiseMedian << "\n";

		float noiseVar = computeVariance(fdp);
		std::cout << "Noisy Variance: " << noiseVar << "\n";

		Mat_<float> outputImage = reconstructImage(noisyImage, r, noiseVar, a, b, true);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);
		imshow("AWA Filter", outputImage);


		float MSE = computeMSE(inputImageFloat, outputImage);
		std::cout << "MSE: " << MSE << "\n";
		std::cout << "255 MSE: " << MSE * 255.0f << "\n";

		float PSNR = computePSNR(MSE);
		std::cout << "PNSR: " << PSNR << "\n";

		// LMMMSE Filtering
		// Mat_<float> outputImageNoisy = reconstructImage(noisyImage, r, noiseVar, a, b, false);
		// imshow("LMMMSE Filter", outputImageNoisy);

		// r = 7
		// a = 1e6
		// b = 10
	}
}