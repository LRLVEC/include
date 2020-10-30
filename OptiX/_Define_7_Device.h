#pragma once
#include <cuda_runtime.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <optix_device.h>
#include <OptiX/_vector_functions.hpp>
#include <curand_kernel.h>

static __forceinline__ __device__ void* uP(unsigned int i0, unsigned int i1)
{
	const unsigned long long uptr = (unsigned long long)(i0) << 32 | i1;
	void* ptr = (void*)(uptr);
	return ptr;
}
static __forceinline__ __device__ void  pP(void* ptr, unsigned int& i0, unsigned int& i1)
{
	const unsigned long long  uptr = (unsigned long long)(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}
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
static __device__ __inline__ float2 random(uint2 st, uint2 xy, unsigned int frame)
{
	float2 a = make_float2(st) / make_float2(xy);
	a.x += float(frame);
	float b(random(a));
	a.y -= float(frame);
	return make_float2(b, random(a));
}
static __device__ __inline__ float2 random1(uint2 st, uint2 xy, unsigned int frame)
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

struct curandStateMini
{
	unsigned int d, v[5];
};
static __device__ __forceinline__ unsigned int curand(curandStateMini* state)
{
	unsigned int t;
	t = (state->v[0] ^ (state->v[0] >> 2));
	state->v[0] = state->v[1];
	state->v[1] = state->v[2];
	state->v[2] = state->v[3];
	state->v[3] = state->v[4];
	state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
	state->d += 362437;
	return state->v[4] + state->d;
}
static __device__ __forceinline__ float curand_uniform(curandStateMini* state)
{
	return _curand_uniform(curand(state));
}
static __device__ __forceinline__ float2 curand_uniform2(curandStateMini* state)
{
	return { _curand_uniform(curand(state)), _curand_uniform(curand(state)) };
}
static __device__ __forceinline__ void getCurandState(curandStateMini* dst, curandState* src)
{
	dst->d = src->d;
#pragma unroll
	for (int c0(0); c0 < 5; ++c0)
		dst->v[c0] = src->v[c0];
}
static __device__ __forceinline__ void setCurandState(curandState* dst, curandStateMini* src)
{
	dst->d = src->d;
#pragma unroll
	for (int c0(0); c0 < 5; ++c0)
		dst->v[c0] = src->v[c0];
}
static __device__ __forceinline__ curandStateMini getCurandStateFromPayload()
{
	curandStateMini tp;
	tp.d = optixGetPayload_2();
	tp.v[0] = optixGetPayload_3();
	tp.v[1] = optixGetPayload_4();
	tp.v[2] = optixGetPayload_5();
	tp.v[3] = optixGetPayload_6();
	tp.v[4] = optixGetPayload_7();
	return tp;
}
static __device__ __forceinline__ void setCurandStateToPayload(curandStateMini* state)
{
	optixSetPayload_2(state->d);
	optixSetPayload_3(state->v[0]);
	optixSetPayload_4(state->v[1]);
	optixSetPayload_5(state->v[2]);
	optixSetPayload_6(state->v[3]);
	optixSetPayload_7(state->v[4]);
}
static __device__ __forceinline__ float3 randomDirectionCosN(float3 normal, float n, curandStateMini* state)
{
	float3 u;
	float3 v;
	float x(powf(curand_uniform(state), 1.0f / (n + 1)));
	float y(2.0 * M_PIf * curand_uniform(state));
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
	return sqrtf(1 - x * x) * (cosf(y) * u + sin(y) * v) + x * normal;
}
static __device__ __forceinline__ float3 randomDirectionCosAngle(float3 normal, float cosAngle, curandStateMini* state)
{
	float3 u;
	float3 v;
	float x(curand_uniform(state) * (1 - cosAngle) + cosAngle);
	float y(2.0 * M_PIf * curand_uniform(state));
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
	return sqrtf(1 - x * x) * (cosf(y) * u + sin(y) * v) + x * normal;
}


static __device__ __forceinline__ float3 transposeMult(float3 r0, float3 r1, float3 r2, float3 a)
{
	float3 t(a.x * r0);
	t += a.y * r1;
	t += a.z * r2;
	return t;
}