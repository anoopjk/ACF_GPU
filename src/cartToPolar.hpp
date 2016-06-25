

#ifndef _cartToPolar_hpp
#define _cartToPolar_hpp

#include <cmath>
#include <float.h>  //DBL_EPSILON
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef PI
#define PI 3.14159265f
#endif

inline
unsigned pow2( unsigned pwr ) {
	return 1 << pwr;
}

static const float atan2_p1 = 0.9997878412794807f*(float)(180/PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180/PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180/PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180/PI);

template<typename T>
static void FastAtan2_32f_xma(const T* Y, const T* X, unsigned* angle, unsigned len, unsigned ang_length)
{    
    const float scale = (float)(1.f/180.f);

    for(unsigned i = 0 ; i < len; ++i )
    {
        float x = static_cast<float>(X[i]), y = static_cast<float>(Y[i]);
        float ax = std::abs(x), ay = std::abs(y);
        float a, c, c2;
		if(ax < 1e-4f && ay < 1e-4f)
		{
			angle[i] = 0.5*pow2(ang_length);
			continue;
		}
        if( ax >= ay )
        {
            c = ay/(ax + (float)DBL_EPSILON);
            c2 = c*c;
            a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }
        else
        {
            c = ax/(ay + (float)DBL_EPSILON);
			c2 = c*c;
            a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }
        if( x < 0 )
            a = 180.f - a;
        if( y < 0 )
            a = 360.f - a;
		float temp = (a*scale);
		angle[i] = (unsigned)round(temp * pow2(ang_length));
    }
	return;
}

template <typename T>
static void Magnitude_32f_xma(const T* x, const T* y, unsigned* mag, unsigned len, unsigned mag_length)
{
    for(unsigned i = 0 ; i < len; i++ )
    {
        float x0 = static_cast<float>(x[i]), y0 = static_cast<float>(y[i]);
        mag[i] = (unsigned)round(std::sqrt(x0*x0 + y0*y0)* pow2(mag_length));
    }
	return;
}

template<typename T>
void cartToPolar_uint( T* dx, T* dy, unsigned* mag, unsigned* ang, const unsigned length, const unsigned mag_fraction, const unsigned ang_fraction)
{
	Magnitude_32f_xma<T>( dx, dy, mag, length, mag_fraction );
	FastAtan2_32f_xma<T>( dy, dx, ang, length, ang_fraction);
	return;
}

/// xma this program assumes the the mag2 is already computed, see acf_detect for more detaileld informaiton
template<typename T>
void cartToPolar_float( T* X, T* Y, float* M, float* O, int len)
{
    const float scale = (float)(1.f/180.f)*PI;

    for(int i = 0 ; i < len; ++i )
    {
		M[i] = sqrt(static_cast<float>(M[i]));
        float x = static_cast<float>(X[i]), y = static_cast<float>(Y[i]);
        float ax = std::abs(x), ay = std::abs(y);
        float a, c, c2;
		/// deal with small values 
		if(ax < 1e-4f && ay < 1e-4f)
		{
			O[i] = 0.5*PI;
			continue;
		}
        if( ax >= ay )
        {
            c = ay/(ax + (float)DBL_EPSILON);
            c2 = c*c;
            a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }
        else
        {
            c = ax/(ay + (float)DBL_EPSILON);
			c2 = c*c;
            a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }

		/// convert from -180:180 to 0-360
        if( x < 0 )
            a = 180.f - a;
        if( y < 0 )
            a = 360.f - a;
		if(y < 0)
			a -= 180.0f;
		float temp = (a*scale);
		O[i] = temp;
    }
	return;
}


#endif
