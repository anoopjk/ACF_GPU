#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iterator>
#include <sys/time.h>
#include <sys/stat.h>
#include <math.h>

#include "cartToPolar.hpp"

using namespace std;


int main(int argc, char** argv)
{

	float *X = new float[8];
	float *Y = new float[8];
	float *M = new float[8];
	float *O = new float[8];
	int i = 0;

	for(;i < 4; ++i)
	{
		X[i] = 0.0f;
		Y[i] = 0.00001 * i*i*1;
	}

	for(;i < 8; ++i)
	{
		X[i] = 0.00001;
		Y[i] = 0.5f * static_cast<float>(i) * pow(-1.0f, static_cast<float>(i));
	}

	cartToPolar_float(X,Y,M,O,8);
	for(i = 0; i < 8; ++i)
	{
		cout << " y= " << left << setw(8) << Y[i] << " x = " << left << setw(8) << X[i] << " O = " << left << setw(8) << O[i] << endl;
	}
	delete[] X,Y,M,O;
	
}

