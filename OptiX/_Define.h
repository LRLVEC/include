//deprecated
#pragma once
#include <optix.h>
#define NOMINMAX
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>

using namespace optix;
static __device__ __inline__ float random(float2 st)
{
	float a(fabsf(sin(dot(st, make_float2(12.9898f, 78.233f))) * 43758.5453123f));
	return a - floorf(a);
}
static __device__ __inline__ float2 random(float2 st, float f)
{
	st.x += f;
	float a(random(st));
	st.y -= f;
	return make_float2(a, random(st));
}
static __device__ __inline__ float2 random(uint2 st, int2 xy, unsigned int frame)
{
	float2 a = make_float2(st) / make_float2(xy);
	a.x += float(frame);
	float b(random(a));
	a.y -= float(frame);
	return make_float2(b, random(a));
}
static __device__ __inline__ float2 random1(uint2 st, int2 xy, unsigned int frame)
{
	float2 a = make_float2(st) / make_float2(xy);
	a.x += sqrtf(frame);
	float b(random(a));
	a.y -= sqrtf(frame);
	return make_float2(b, random(a));
}
static __device__ __inline__ float2 randomCircle(float2 seed)
{
	seed.x = sqrtf(seed.x);
	seed.y *= 2 * M_PIf;
	return make_float2(cosf(seed.y) * seed.x, sinf(seed.y) * seed.x);
}
static __device__ __inline__ float3 randomNormal(float3 n, float2 seed)
{
	float3 u;
	float3 v;
	seed.x = random(seed);
	seed.y = 2.0 * M_PIf * random(seed);
	if (fabsf(n.x) > 0.7f)
	{
		float s = sqrtf(1 - n.y * n.y);
		u = make_float3(-n.z, 0, n.x) / s;
		v = make_float3(n.x * n.y / s, -s, n.y * n.z / s);
	}
	else
	{
		float s = sqrtf(1 - n.x * n.x);
		u = make_float3(0, n.z, -n.y) / s;
		v = make_float3(-s, n.x * n.y / s, n.x * n.z / s);
	}
	float sinTheta = sqrtf(1 - seed.x * seed.x);
	return sinTheta * cosf(seed.y) * u + sinTheta * sin(seed.y) * v + seed.x * n;

}
static __device__ __inline__ float3 randomDirectionCosN(float3 normal, float n, float2 seed)
{
	float3 u;
	float3 v;
	seed.x = random(seed);
	seed.y = 2.0 * M_PIf * random(seed);
	if (fabsf(normal.x) > 0.7f)
	{
		float s = sqrtf(1 - normal.y * normal.y);
		u = make_float3(-normal.z, 0, normal.x) / s;
		v = make_float3(normal.x * normal.y / s, -s, normal.y * normal.z / s);
	}
	else
	{
		float s = sqrtf(1 - normal.x * normal.x);
		u = make_float3(0, normal.z, -normal.y) / s;
		v = make_float3(-s, normal.x * normal.y / s, normal.x * normal.z / s);
	}
	seed.x = powf(seed.x, 1.0f / (n + 1));
	return sqrtf(1 - seed.x * seed.x) * (cosf(seed.y) * u + sin(seed.y) * v) + seed.x * normal;
}
//(1/3)x^3+x=a
static __device__ __inline__ float solveX3X(float a)
{
	a *= 3;
	float t(0.79370052598409973738f * (a + sqrtf(4 + a * a)));
	return t - 1 / t;
}
static __device__ __inline__ float4 randomScatter(float3 direction, float t, float k, float3 seed)
{

	seed.z = logf(expm1f(t * k) * random(make_float2(seed.x, seed.y)) + 1) / k;
	seed.x = solveX3X(2.6666666666666666667 * random(make_float2(seed.y, seed.z)) - 1.3333333333333333333);
	seed.y = 2.0 * M_PIf * random(make_float2(seed.x, seed.z));
	float3 u;
	float3 v;
	if (fabsf(direction.x) > 0.7f)
	{
		float s = sqrtf(1 - direction.y * direction.y);
		u = make_float3(-direction.z, 0, direction.x) / s;
		v = make_float3(direction.x * direction.y / s, -s, direction.y * direction.z / s);
	}
	else
	{
		float s = sqrtf(1 - direction.x * direction.x);
		u = make_float3(0, direction.z, -direction.y) / s;
		v = make_float3(-s, direction.x * direction.y / s, direction.x * direction.z / s);
	}
	return make_float4(sqrtf(1 - seed.x * seed.x) * (cosf(seed.y) * u + sin(seed.y) * v) + seed.x * direction, seed.z);
}
namespace Define
{
	/*template<unsigned int N>static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1)
	{
		unsigned int v0 = val0;
		unsigned int v1 = val1;
		unsigned int s0 = 0;
		for (unsigned int n = 0; n < N; n++)
		{
			s0 += 0x9e3779b9;
			v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
			v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
		}
		return v0;
	}
	static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev)//[0, 2^24)
	{
		const unsigned int LCG_A = 1664525u;
		const unsigned int LCG_C = 1013904223u;
		prev = (LCG_A * prev + LCG_C);
		return prev & 0x00FFFFFF;
	}
	static __host__ __device__ __inline__ unsigned int lcg2(unsigned int& prev)
	{
		prev = (prev * 8121 + 28411) % 134456;
		return prev;
	}
	static __host__ __device__ __inline__ float random(unsigned int& prev)
	{
		return ((float)lcg(prev) / (float)0x01000000);
	}
	static __host__ __device__ __inline__ unsigned int rotSeed(unsigned int seed, unsigned int frame)
	{
		return seed ^ frame;
	}*/
	__device__ float3 const scatterRatio = { 0.3208506003309145f,0.9224630443030504f ,1.756686355366035f };

	struct Trans
	{
		Matrix3x4 trans;
		float3 r0;
		float z0;
	};
	struct TransDepth
	{
		Matrix3x4 trans;
		float3 r0;
		float V;
		float P;
		float D;
	};
}